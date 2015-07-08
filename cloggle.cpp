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
#include <chrono>
#include <random>
#include <set>

extern "C" {
int CLID = 0;
#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ulong unsigned long long
#define kernel
#define constant
#define global
int get_global_id(int n) { return CLID; }
int get_local_id(int n) { return 0; }
int get_global_size(int n) { return n; }
#include "res/cloggle.cl"
}

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const int NUM_CL_THREADS    = 1024*8;
const bool SOFTWARE_GRIND   = false;
const int NUM_CL_ITERATIONS = 1000000;

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
            for (int j = curN; j <= MAX_NEIGHBORS; j++) { 
                neighbors[i*(MAX_NEIGHBORS + 1) + j] = -1;
            }
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

    void createTrie(const std::string& dict) {
        size_t pos = 0;
        while (dict[pos]) {
            pos += insertWord(dict.c_str() + pos);
            while (isspace(dict[pos])) pos++;
        }
    }
};

std::string loadFile(const char* fileName) {
    std::ifstream f(fileName);
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();
    return buffer.str();
}

int main() {
    std::string dict = loadFile("res/words.txt");

    //  load and preprocess dice (move the unique ones to front)
    std::string dice = loadFile("res/dice.txt");
    std::vector<unsigned char> diceNum(BOARD_SIZE);
    dice.erase(std::remove(dice.begin(), dice.end(), '\n'), dice.end());
    for (int i = 0; i < BOARD_SIZE; i++) {
        std::set<char> duniq;
        std::vector<char> drest;
        for (int j = 0; j < DIE_FACES; j++) {
            char d = dice[j + i*DIE_FACES];
            if (duniq.find(d) != duniq.end()) {
                drest.push_back(d);
            } else {
                duniq.insert(d);
            }
        }
        std::copy(duniq.begin(), duniq.end(), dice.begin() + i*DIE_FACES);
        std::copy(drest.begin(), drest.end(), dice.begin() + i*DIE_FACES + duniq.size());
        diceNum[i] = (unsigned char)duniq.size();
    }

    Boggle boggle;
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
    
    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    cl_device_id device_id;
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);   
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    static_assert(sizeof(BoardCL) == 32, "BoardCL sizeof must be 32");

    cl_mem d_trie_nodes         = clCreateBuffer(context, CL_MEM_READ_ONLY, nodes.size()*sizeof(TrieNodeCL), NULL, &ret);
    cl_mem d_trie_edge_labels   = clCreateBuffer(context, CL_MEM_READ_ONLY, edgeLabels.size()*sizeof(char), NULL, &ret);
    cl_mem d_trie_edge_targets  = clCreateBuffer(context, CL_MEM_READ_ONLY, edgeTargets.size()*sizeof(unsigned short), NULL, &ret);
    cl_mem d_dice               = clCreateBuffer(context, CL_MEM_READ_ONLY, BOARD_SIZE*DIE_FACES*sizeof(unsigned char), NULL, &ret);
    cl_mem d_num_dice           = clCreateBuffer(context, CL_MEM_READ_ONLY, BOARD_SIZE*DIE_FACES*sizeof(unsigned char), NULL, &ret);
    cl_mem d_cell_neighbors     = clCreateBuffer(context, CL_MEM_READ_ONLY, (BOARD_SIZE*(MAX_NEIGHBORS + 1))*sizeof(char), NULL, &ret);
    cl_mem d_boards             = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_CL_THREADS*sizeof(BoardCL), NULL, &ret);

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
        std::cerr << log << std::endl << std::flush;
    }

    cl_kernel kern_grind = clCreateKernel(program, "grind", &ret);
    ret = clSetKernelArg(kern_grind, 0, sizeof(cl_mem), (void*)&d_trie_nodes);
    ret = clSetKernelArg(kern_grind, 1, sizeof(int),    (void*)&numNodes);
    ret = clSetKernelArg(kern_grind, 2, sizeof(cl_mem), (void*)&d_trie_edge_labels);
    ret = clSetKernelArg(kern_grind, 3, sizeof(cl_mem), (void*)&d_trie_edge_targets);
    ret = clSetKernelArg(kern_grind, 4, sizeof(cl_mem), (void*)&d_dice);
    ret = clSetKernelArg(kern_grind, 5, sizeof(cl_mem), (void*)&d_num_dice);
    ret = clSetKernelArg(kern_grind, 6, sizeof(cl_mem), (void*)&d_cell_neighbors);
    ret = clSetKernelArg(kern_grind, 7, sizeof(cl_mem), (void*)&d_boards);

    std::vector<BoardCL> boards(NUM_CL_THREADS);
    BoardCL bestBoard = { 0, MAX_PLATEAU_AGE };
    std::fill(boards.begin(), boards.end(), bestBoard);

    size_t globalWorkSize[] = {NUM_CL_THREADS};

    ret = clEnqueueWriteBuffer(command_queue, d_trie_nodes, CL_TRUE, 0, nodes.size()*sizeof(TrieNodeCL), &clNodes[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_labels, CL_TRUE, 0, edgeLabels.size()*sizeof(char), &edgeLabels[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_targets, CL_TRUE, 0, edgeTargets.size()*sizeof(unsigned short), &edgeTargets[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_dice, CL_TRUE, 0, BOARD_SIZE*DIE_FACES*sizeof(char), &dice[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_num_dice, CL_TRUE, 0, BOARD_SIZE*DIE_FACES*sizeof(char), &diceNum[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_cell_neighbors, CL_TRUE, 0, (BOARD_SIZE*(MAX_NEIGHBORS + 1))*sizeof(char), &boggle.neighbors, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_boards, CL_TRUE, 0, NUM_CL_THREADS*sizeof(BoardCL), &boards[0], 0, NULL, NULL);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CL_ITERATIONS; i++) {
        if (!SOFTWARE_GRIND) {
          ret = clEnqueueNDRangeKernel(command_queue, kern_grind, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
          ret = clEnqueueReadBuffer(command_queue, d_boards, CL_FALSE, 0, NUM_CL_THREADS*sizeof(BoardCL), &boards[0], 0, NULL, NULL);
        } else {
          for (int j = 0; j < NUM_CL_THREADS; j++) {
            CLID = j;
            grind((const TrieNodeCL*)&clNodes[0], (int)clNodes.size(), &edgeLabels[0], &edgeTargets[0], dice.c_str(), &diceNum[0],
              (const char*)&boggle.neighbors, &boards[0]);
          }
        }
        auto mit = std::max_element(boards.begin(), boards.end(), 
          [](const BoardCL& a, const BoardCL& b) { return a.score < b.score; });
        const BoardCL& board = *mit;
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("%d. step: %d, time: %dms, ", board.score,
          i, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        printf("board: ");
        for (int j = 0; j < BOARD_SIZE; j++) printf("%c", dice[board.cells[j]]);
        printf(" dice: ");
        for (int j = 0; j < BOARD_SIZE; j++) printf("%d ", board.cells[j] / DIE_FACES);
        
        if (board.score >= bestBoard.score) {
          bestBoard = board;
        } else {
          printf(" best: %d ", bestBoard.score);
          for (int j = 0; j < BOARD_SIZE; j++) printf("%c", dice[bestBoard.cells[j]]);
        }
        
        printf("\n");
    }

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kern_grind);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_trie_nodes);
    ret = clReleaseMemObject(d_trie_edge_labels);
    ret = clReleaseMemObject(d_trie_edge_targets);
    ret = clReleaseMemObject(d_cell_neighbors);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}