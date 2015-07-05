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

#include "constants.h"

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

const int NUM_CL_THREADS = 250;
const int NUM_CL_ITERATIONS = 100000;
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

std::vector<unsigned short> TEST_SCORES = {
639,
46,
611,
529,
601,
45,
617,
521,
202,
609
};

void printPlatformInfo(cl_platform_id platform_id) {
  char* info;
  size_t infoSize;
  const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
  const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
      CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
  const int attributeCount = sizeof(attributeNames) / sizeof(char*);
 
  for (int j = 0; j < attributeCount; j++) {
    // get platform attribute value size
    clGetPlatformInfo(platform_id, attributeTypes[j], 0, NULL, &infoSize);
    info = (char*) malloc(infoSize);
 
    // get platform attribute value
    clGetPlatformInfo(platform_id, attributeTypes[j], infoSize, info, NULL);
 
    printf("  %-11s: %s\n", attributeNames[j], info);
    free(info);
  }
}

void printDeviceInfo(cl_device_id device) {
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;
    
    // print device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);
    free(value);
 
    // print hardware device version
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
    printf(" Hardware version: %s\n", value);
    free(value);
 
    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
    printf(" Software version: %s\n", value);
    free(value);
 
    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf(" OpenCL C version: %s\n", value);
    free(value);
 
    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf(" Parallel compute units: %d\n", maxComputeUnits);

    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    printf(" Max work group size: %d\n", maxWorkGroupSize);

    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
            sizeof(workitem_size), &workitem_size, NULL);
    printf(" Work item size: %d\n", workitem_size[0]);
}

int main() {
    std::string dict = loadFile("res/words.txt");
    std::string dice = loadFile("res/dice.txt");
    
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
    const int genePoolSize = GENE_POOL_SIZE;
    
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

    std::vector<unsigned short> cpu_scores(GENE_POOL_SIZE);
    std::string genes = TEST_STRINGS;

    eval((const Node*)&clNodes[0], (int)clNodes.size(), &edgeLabels[0], &edgeTargets[0], dice.c_str(), 
      (const char*)&boggle.neighbors, (char*)genes.c_str(), &cpu_scores[0], genePoolSize);

    
    cl_platform_id platform_ids[8] = {};
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(8, platform_ids, &ret_num_platforms);
    for (cl_uint i = 0; i < ret_num_platforms; i++) {
        printf("Platform %d:", i);
        printPlatformInfo(platform_ids[i]);
    }

    cl_platform_id platform_id = platform_ids[0];
    cl_device_id device_ids[8] = {};
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 8, device_ids, &ret_num_devices);
    for (cl_uint i = 0; i < ret_num_devices; i++) {
        printf("Device %d:", i);
        printDeviceInfo(device_ids[i]);
    }

    cl_device_id device_id;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);   
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

    cl_mem d_trie_nodes         = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nodes.size()*sizeof(TrieNodeCL), NULL, &ret);
    cl_mem d_trie_edge_labels   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, edgeLabels.size()*sizeof(char), NULL, &ret);
    cl_mem d_trie_edge_targets  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, edgeTargets.size()*sizeof(unsigned short), NULL, &ret);
    cl_mem d_dice               = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BOARD_SIZE*(DIE_FACES + 1)*sizeof(char), NULL, &ret);
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
        std::cerr << log << std::endl << std::flush;
    }

    cl_kernel kern = clCreateKernel(program, "eval", &ret);

    ret = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&d_trie_nodes);
    ret = clSetKernelArg(kern, 1, sizeof(int),    (void*)&numNodes);
    ret = clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&d_trie_edge_labels);
    ret = clSetKernelArg(kern, 3, sizeof(cl_mem), (void*)&d_trie_edge_targets);
    ret = clSetKernelArg(kern, 4, sizeof(cl_mem), (void*)&d_dice);
    ret = clSetKernelArg(kern, 5, sizeof(cl_mem), (void*)&d_cell_neighbors);
    ret = clSetKernelArg(kern, 6, sizeof(cl_mem), (void*)&d_gene_pool);
    ret = clSetKernelArg(kern, 7, sizeof(cl_mem), (void*)&d_scores);
    ret = clSetKernelArg(kern, 8, sizeof(int),    (void*)&genePoolSize);


    ret = clEnqueueWriteBuffer(command_queue, d_trie_nodes,         CL_TRUE, 0, nodes.size()*sizeof(TrieNodeCL), &clNodes[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_labels,   CL_TRUE, 0, edgeLabels.size()*sizeof(char), &edgeLabels[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_trie_edge_targets,  CL_TRUE, 0, edgeTargets.size()*sizeof(unsigned short), &edgeTargets[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_dice,               CL_TRUE, 0, BOARD_SIZE*(DIE_FACES + 1)*sizeof(char), &dice[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_cell_neighbors,     CL_TRUE, 0, (BOARD_SIZE*(MAX_NEIGHBORS + 1))*sizeof(char), &boggle.neighbors, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_gene_pool,          CL_TRUE, 0, (BOARD_SIZE*GENE_POOL_SIZE)*sizeof(char), TEST_STRINGS, 0, NULL, NULL);

    ret = clFinish(command_queue);

    ret = clEnqueueTask(command_queue, kern, 0, NULL, NULL);

    std::vector<unsigned short> cl_scores(GENE_POOL_SIZE);
    ret = clEnqueueReadBuffer(command_queue, d_scores, CL_TRUE, 0, GENE_POOL_SIZE*sizeof(unsigned short), &cl_scores[0], 0, NULL, NULL);

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);

    assert(cl_scores == TEST_SCORES);
    assert(cpu_scores == cl_scores);


    cl_kernel kern_grind = clCreateKernel(program, "grind", &ret);
    cl_mem d_best_boards = clCreateBuffer(context, CL_MEM_READ_WRITE,  (BOARD_SIZE*NUM_CL_THREADS)*sizeof(char), NULL, &ret);
    cl_mem d_best_scores = clCreateBuffer(context, CL_MEM_READ_ONLY,  NUM_CL_THREADS*sizeof(unsigned short), NULL, &ret);

    ret = clSetKernelArg(kern_grind, 0, sizeof(cl_mem), (void*)&d_trie_nodes);
    ret = clSetKernelArg(kern_grind, 1, sizeof(int),    (void*)&numNodes);
    ret = clSetKernelArg(kern_grind, 2, sizeof(cl_mem), (void*)&d_trie_edge_labels);
    ret = clSetKernelArg(kern_grind, 3, sizeof(cl_mem), (void*)&d_trie_edge_targets);
    ret = clSetKernelArg(kern_grind, 4, sizeof(cl_mem), (void*)&d_dice);
    ret = clSetKernelArg(kern_grind, 5, sizeof(cl_mem), (void*)&d_cell_neighbors);
    ret = clSetKernelArg(kern_grind, 6, sizeof(cl_mem), (void*)&d_best_boards);
    ret = clSetKernelArg(kern_grind, 7, sizeof(cl_mem), (void*)&d_best_scores);

    size_t globalWorkSize[] = {NUM_CL_THREADS};

    std::vector<char> boards(BOARD_SIZE*NUM_CL_THREADS);
    std::vector<unsigned short> scores(NUM_CL_THREADS);
    std::string best_board;

    //  randomly init the boards
    std::mt19937 rnd(1234567);
    char* pboards = &boards[0];
    for (int i = 0; i < NUM_CL_THREADS; i++)
    {
      for (int j = 0; j < BOARD_SIZE; j++) {
        boards[j + i*BOARD_SIZE] = dice[rnd()%DIE_FACES + (DIE_FACES + 1)*j];
      }
      std::shuffle(pboards + i*BOARD_SIZE, pboards + (i + 1)*BOARD_SIZE, rnd);
    }

    //  evaluate the initial boards
    eval((const Node*)&clNodes[0], (int)clNodes.size(), &edgeLabels[0], &edgeTargets[0], dice.c_str(), 
      (const char*)&boggle.neighbors, &boards[0], &scores[0], NUM_CL_THREADS);

    unsigned short best_score = 0;

    ret = clEnqueueWriteBuffer(command_queue, d_best_scores, CL_TRUE, 0, NUM_CL_THREADS*sizeof(unsigned short), &scores[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_best_boards, CL_TRUE, 0, NUM_CL_THREADS*BOARD_SIZE*sizeof(unsigned char), &boards[0], 0, NULL, NULL);

    bool software = false;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CL_ITERATIONS; i++) {
        unsigned int seed = rnd();
        
        if (!software) {
          ret = clSetKernelArg(kern_grind, 8, sizeof(unsigned int), (void*)&seed);
          ret = clEnqueueNDRangeKernel(command_queue, kern_grind, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
          ret = clEnqueueReadBuffer(command_queue, d_best_scores, CL_TRUE, 0, NUM_CL_THREADS*sizeof(unsigned short), &scores[0], 0, NULL, NULL);
          ret = clEnqueueReadBuffer(command_queue, d_best_boards, CL_TRUE, 0, NUM_CL_THREADS*BOARD_SIZE*sizeof(unsigned char), &boards[0], 0, NULL, NULL);
        }
        else {
          for (int j = 0; j < NUM_CL_THREADS; j++) {
            CLID = j;
            grind((const Node*)&clNodes[0], (int)clNodes.size(), &edgeLabels[0], &edgeTargets[0], dice.c_str(),
              (const char*)&boggle.neighbors, (char*)&boards[0], &scores[0], seed);
          }
        }
        auto mit = std::max_element(scores.begin(), scores.end());
        if (*mit > best_score) {
            best_score = *mit;
            best_board = std::string(&boards[BOARD_SIZE*(mit - scores.begin())], BOARD_SIZE);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("step: %d, score: %d, board: %s, best: %d %s, time: %dms\n", i, *mit, 
          std::string(&boards[BOARD_SIZE*(mit - scores.begin())], BOARD_SIZE).c_str(),
          best_score,
          best_board.c_str(),
          std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);

    ret = clReleaseKernel(kern);
    ret = clReleaseKernel(kern_grind);

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