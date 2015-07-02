#define GENE_POOL_SIZE 10
#define MAX_NEIGHBORS 8
#define MAX_WORD_LEN 16
#define BOARD_SIDE 5
#define BOARD_SIZE BOARD_SIDE*BOARD_SIDE

#define MAX_TRIE_SIZE 10000

typedef struct
{
    uchar   score;              //  node score, assuming terminal if non-0
    uchar   num_edges;          //  number of outgoing edges
    ushort  edges_offset;       //  offset in the edge label/target arrays
} TrieNode;

kernel void grind(
    constant const TrieNode*  g_trie_nodes, 
    int                       g_num_trie_nodes,
    constant const char*      g_trie_edge_labels,
    constant const ushort*    g_trie_edge_targets,
    constant const char*      g_cell_neighbors,
    global char*              g_gene_pool,
    global ushort*            g_scores)
{
    //  evaluate the boards
    for (int i = 0 ; i < 1/*GENE_POOL_SIZE*/; i++) {
        uchar visited_faces[BOARD_SIZE] = {};
        uchar visited_nodes[MAX_TRIE_SIZE] = {};
        ushort score = 0;

        for (int j = 0; j < 1/*BOARD_SIZE*/; j++) {
            //  "recursively" depth-first search inside the trie and bard in parallel
            uchar cell_stack[MAX_WORD_LEN] = {};
            ushort node_stack[MAX_WORD_LEN] = {};
            uchar cur_neighbor[MAX_WORD_LEN] = {};
            
            int depth = 0;
            cell_stack[0] = j;
            node_stack[0] = 0;

            do {
                int cell = cell_stack[depth];
                int node = node_stack[depth];

                bool backtrack = true;
                printf("Depth: %d, cell: %d, node: %d\n", depth, cell, node);
                char c = g_gene_pool[cell + BOARD_SIZE*i];
                //  find the outgoing edge
                int child_node = -1;
                for (int k = 0, e = (int)g_trie_nodes[node].num_edges; k < e; k++) {
                    int edge_offs = g_trie_nodes[node].edges_offset + k;
                    if (g_trie_edge_labels[edge_offs] == c) {
                        child_node = g_trie_edge_targets[edge_offs];
                        break;
                    }
                }
                printf("  Char: %c, idx: %d\n", c, child_node);
                    
                if (child_node >= 0) {
                    //  the prefix is in the dictionary
                    if (g_trie_nodes[child_node].score > 0) {
                        //  the prefix is also a full word
                        score += (ushort)g_trie_nodes[child_node].score;
                        printf(" SCORE: %d", (ushort)g_trie_nodes[child_node].score);
                    }
                    //  go down, depth-first
                    visited_faces[cell] = 1;
                    int neighbor_cell = -1;
                    do {
                        neighbor_cell = g_cell_neighbors[cur_neighbor[depth] + (MAX_NEIGHBORS + 1)*cell];
                        cur_neighbor[depth]++;
                    } while (neighbor_cell >= 0 && visited_faces[neighbor_cell]);
                    if (neighbor_cell > -1) {
                        backtrack = false;
                        depth++;
                        cell_stack[depth] = neighbor_cell;
                        node_stack[depth] = child_node;
                    }
                } 
                
                if (backtrack) {
                    visited_faces[cell] = 0;
                    cur_neighbor[depth] = 0;
                    depth--;
                    printf("Backtrack to: %d\n", cell);
                }
            } while (depth > 0);
        }
        g_scores[i] = score;
    }
}