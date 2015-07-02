#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>

#include "constants.h"

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const int SCORE_LOOKUP[] = {0, 0, 0, 0, 1, 2, 3, 5, 11};
const int OFFS[][MAX_NEIGHBORS] = {
        {0, -1}, {1, -1}, {1, 0}, {1, 1}, 
        {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}};

struct Boggle {
    char neighbors[BOARD_SIZE*(MAX_NEIGHBORS + 1)];

    Boggle() {
        createNeighbors();
    }

    void createNeighbors() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            int x = i % BOARD_SIDE, y = i / BOARD_SIDE;
            int curN = 0;
            for (int j = 0; j < MAX_NEIGHBORS; j++) {
                int cx = x + OFFS[j][0], cy = y + OFFS[j][1];
                bool isInside = (0 <= cx && cx < BOARD_SIDE && 0 <= cy && cy < BOARD_SIDE);
                if (isInside) {
                    neighbors[i*(MAX_NEIGHBORS + 1) + curN] = (cx + cy*BOARD_SIDE);
                    curN++;
                }
            }
            neighbors[i*(MAX_NEIGHBORS + 1) + curN] = -1;
        } 
    }

    inline static int score(int worldLen) {
        return SCORE_LOOKUP[std::min(8, worldLen)];
    }
};

struct TrieNode {
    bool terminal;
    int score;
    int index;
    std::map<char, TrieNode*> edges;

    TrieNode() : terminal(false), score(0) {}
    ~TrieNode() {
        for (auto& e: edges) delete e.second;
    }

    int insertWord(const char* word, int pos = 0) {
        int c = word[pos];
        if (c == 0 || isspace(c)) {
            // end of the word
            terminal = true;
            score = Boggle::score(pos);
            return pos;
        }
        //  normalize the character
        c = tolower(c);
        bool skipWord = false;
        if (c == 'q') {
            if (tolower(word[pos + 1]) == 'u') pos++;
            else skipWord = true;
        }
        if (c < 'a' || c > 'z') skipWord = true;
        if (skipWord) {
            while (!isspace(word[pos])) pos++;
            return pos;
        }
        //  append the next child
        TrieNode* child;
        if (edges.find(c) == edges.end()) {
            child = new TrieNode(); 
            edges[c] = child;
        } else child = edges[c];
        return child->insertWord(word, pos + 1);
    }

    //  depth-first trie traversal with given visitor function
    template <typename TF>
    void traverseDF(TF visit) {
        visit(*this);
        for (auto& e : edges) e.second->traverseDF(visit);
    }

    void dumpToJSON(std::ostream& os) {
        os << "{";
        //os << "\"terminal\":" << (terminal ? "\"true\"" : "\"false\"") << ",";
        os << "\"score\":" << score << ",";
        os << "\"edges\": {"; 
        for(auto it = edges.begin(); it != edges.end(); ++it) {
            if (it != edges.begin()) os << ",";
            os << "\"" << (char)it->first << "\":";
            it->second->dumpToJSON(os);
        }
        os << "}}";
    }

    void createTrie(const std::string& dict) {
        size_t pos = 0;
        while (dict[pos]) {
            pos += insertWord(dict.c_str() + pos);
            while (isspace(dict[pos])) pos++;
        }
    }
};

struct TrieNodeCL
{
    unsigned char   score;         //  node score, assuming terminal if non-0
    unsigned char   num_edges;     //  number of outgoing edges
    unsigned short  edges_offset;  //  offset in the edge label/target arrays
};


#define MEM_SIZE (128)

std::string loadFile(const char* fileName) {
    std::ifstream f(fileName);
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();
    return buffer.str();
}

const char* TEST_STRINGS = 
"herdotceinrntsveaioeglprq"
"tteetteenntteaasregrbrere"
"qsndviteeignrtsceainrplco"
"imapeonerlcstairiencbdret"
"renotvsticieraldgnephtcdb"
"sgecaaremecgntdoyspjnoicd"
"trplgeoiaevstnrniectodreq"
"dhreqtecatsntsliaeioplrgn"
"gdtteserethstanraigrtknee"
"rlipbegarohnitetcesneradi"
;

const unsigned short TEST_SCORES[] = {
639,
46,
611,
529,
601,
45,
617,
512,
202,
609
};

int main() {
    std::string dict = loadFile("res/words.txt");
    TrieNode trie;
    trie.createTrie(dict);

    //  enumerate the nodes
    std::vector<TrieNode*> nodes;
    trie.traverseDF([&nodes](TrieNode& n) { 
        n.index = (int)nodes.size();
        nodes.push_back(&n); 
    });
    const int numNodes = (int)nodes.size();
    
    //  create device-side data structures
    std::vector<char> edgeLabels;
    std::vector<unsigned short> edgeTargets;
    std::vector<TrieNodeCL> clNodes;
    for (auto& node : nodes) { 
        TrieNodeCL ncl;
        ncl.score = node->score;
        ncl.num_edges = (unsigned char)node->edges.size();
        ncl.edges_offset = (unsigned int)edgeLabels.size();
        clNodes.push_back(ncl);

        for (auto& edge : node->edges) {
            edgeLabels.push_back(edge.first);
            assert(edgeTargets.size() < (1 << sizeof(unsigned short)*8));
            edgeTargets.push_back(edge.second->index);
        }
    }

    Boggle boggle;
    
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);   
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem d_trie_nodes         = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nodes.size()*sizeof(TrieNodeCL), NULL, &ret);
    cl_mem d_trie_edge_labels   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, edgeLabels.size()*sizeof(char), NULL, &ret);
    cl_mem d_trie_edge_targets  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, edgeTargets.size()*sizeof(unsigned short), NULL, &ret);
    cl_mem d_cell_neighbors     = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (BOARD_SIZE*(MAX_NEIGHBORS + 1))*sizeof(char), NULL, &ret);
    
    cl_mem d_gene_pool          = clCreateBuffer(context, CL_MEM_READ_WRITE,  (BOARD_SIZE*GENE_POOL_SIZE)*sizeof(char), NULL, &ret);
    cl_mem d_scores             = clCreateBuffer(context, CL_MEM_READ_ONLY,  GENE_POOL_SIZE*sizeof(unsigned short), NULL, &ret);


    std::string kernelCode = loadFile("res/cloggle.cl");
    const char* source_str = kernelCode.c_str();
    size_t source_size = kernelCode.size();
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::string log;
        log.reserve(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        std::cout << log << std::endl << std::flush;
    }

    cl_kernel kernel = clCreateKernel(program, "grind", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_trie_nodes);
    ret = clSetKernelArg(kernel, 1, sizeof(int),    (void*)&numNodes);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_trie_edge_labels);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&d_trie_edge_targets);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&d_cell_neighbors);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&d_gene_pool);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&d_scores);


    ret = clEnqueueWriteBuffer(command_queue, d_trie_nodes,         CL_TRUE, 0, nodes.size()*sizeof(TrieNodeCL), &clNodes[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_labels,   CL_TRUE, 0, edgeLabels.size()*sizeof(char), &edgeLabels[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_targets,  CL_TRUE, 0, edgeTargets.size()*sizeof(unsigned short), &edgeTargets[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_cell_neighbors,     CL_TRUE, 0, (BOARD_SIZE*(MAX_NEIGHBORS + 1))*sizeof(char), &boggle.neighbors, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_gene_pool,          CL_TRUE, 0, (BOARD_SIZE*GENE_POOL_SIZE)*sizeof(char), TEST_STRINGS, 0, NULL, NULL);

    ret = clFinish(command_queue);

    ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    unsigned short score;
    char best_board[BOARD_SIZE];
    ret = clEnqueueReadBuffer(command_queue, d_scores, CL_TRUE, 0, sizeof(unsigned short), &score, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_gene_pool, CL_TRUE, 0, BOARD_SIZE*sizeof(unsigned char), best_board, 0, NULL, NULL);

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_trie_nodes);
    ret = clReleaseMemObject(d_trie_edge_labels);
    ret = clReleaseMemObject(d_trie_edge_targets);
    ret = clReleaseMemObject(d_cell_neighbors);
    ret = clReleaseMemObject(d_gene_pool);
    ret = clReleaseMemObject(d_scores);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}