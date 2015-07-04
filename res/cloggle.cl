#define GENE_POOL_SIZE 10
#define MAX_NEIGHBORS 8
#define MAX_WORD_LEN 16
#define BOARD_SIDE 5
#define DIE_FACES 6
#define BOARD_SIZE BOARD_SIDE*BOARD_SIDE

#define MAX_EPOCH 10
#define MAX_TRIE_SIZE 1100

//  trie node
typedef struct
{
  uchar   score;          //  node score, assuming terminal if non-0
  uchar   num_edges;      //  number of outgoing edges
  ushort  edges_offset;   //  offset in the edge label/target arrays
} Node;


//  poor-man random number generator (stolen from Java library implementation)
ulong rnd(ulong* seed) { 
  return *seed = ((*seed)*0x5DEECE66DL + 0xBL)&(((ulong)1 << 48) - 1); 
}

//  shuffle array in-place (Fischer-Yites)
void shuffle_board(char* board, ulong* seed) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    int swap_with = rnd(seed)%(BOARD_SIZE - i);
    char tmp = board[swap_with];
    board[swap_with] = board[i];
    board[i] = tmp;
  }
}

//  creates a random board from the set of dice
void make_random_board(char* board, constant const char* g_dice, ulong* seed) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    board[i] = g_dice[i*(DIE_FACES + 1) + rnd(seed)%DIE_FACES];
  }
  shuffle_board(board, seed);
}

//  evaluates the board score
ushort eval_board(
  constant const Node*    g_trie_nodes, 
  int                     g_num_trie_nodes,
  constant const char*    g_trie_edge_labels,
  constant const ushort*  g_trie_edge_targets,
  constant const char*    g_dice,
  constant const char*    g_cell_neighbors,
  const char*             board
) {
  uchar visited_nodes[MAX_TRIE_SIZE] = {};
  ushort score = 0;

  for (int j = 0; j < BOARD_SIZE; j++) {
    //  "recursively" depth-first search inside the trie and bard in parallel
    uchar visited_faces[BOARD_SIZE] = {};

    uchar cell_stack[MAX_WORD_LEN];
    ushort node_stack[MAX_WORD_LEN];
    uchar cur_neighbor_stack[MAX_WORD_LEN];
      
    int depth = 0;

    cell_stack[0] = j;
    node_stack[0] = 0;
    cur_neighbor_stack[0] = 0;
    visited_faces[0] = 0;

    do {
      int cell = cell_stack[depth];
      int node = node_stack[depth];
      bool backtrack = true;
      if (cur_neighbor_stack[depth] == 0) {
        //  find the outgoing edge, corresponding to the current cell
        char c = board[cell];
        int edge_offs = g_trie_nodes[node].edges_offset;
        int num_edges = (int)g_trie_nodes[node].num_edges;
        node = -1;
        for (int k = 0; k < num_edges; k++) {
          if (g_trie_edge_labels[edge_offs + k] == c) {
            node = g_trie_edge_targets[edge_offs + k];
            //  the prefix also may be a full word, add the score
            score += (ushort)g_trie_nodes[node].score*(1 - (visited_nodes[node/8] >> (node%8))&1);
            visited_nodes[node/8] |= (1 << (node%8));
            break;
          }
        }
      }
          
      if (node >= 0) {
        //  go down, depth-first
        node_stack[depth] = node;
        visited_faces[cell] = 1;
        int neighbor_cell = -1;
        do {
          neighbor_cell = g_cell_neighbors[cur_neighbor_stack[depth] + (MAX_NEIGHBORS + 1)*cell];
          cur_neighbor_stack[depth]++;
        } while (neighbor_cell >= 0 && visited_faces[neighbor_cell]);

        if (neighbor_cell > -1) {
          backtrack = false;
          depth++;
          cell_stack[depth] = neighbor_cell;
          node_stack[depth] = node;
          cur_neighbor_stack[depth] = 0;
        }
      } 
        
      if (backtrack) {
        visited_faces[cell] = 0;
        depth--;
      }
    } while (depth >= 0);
  }
  return score;
}

//  board evaluation kernel
kernel void eval(
  constant const Node*    g_trie_nodes, 
  int                     g_num_trie_nodes,
  constant const char*    g_trie_edge_labels,
  constant const ushort*  g_trie_edge_targets,
  constant const char*    g_dice,
  constant const char*    g_cell_neighbors,
  global char*            g_gene_pool,
  global ushort*          g_scores)
{
  char board[BOARD_SIZE];
  for (int i = 0; i < GENE_POOL_SIZE; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      board[j] = g_gene_pool[i*BOARD_SIZE + j];
    }

    g_scores[i] = eval_board(g_trie_nodes, g_num_trie_nodes, g_trie_edge_labels, g_trie_edge_targets,
                             g_dice, g_cell_neighbors, board);
  }
}

//  the grinding kernel 
kernel void grind(
  constant const Node*    g_trie_nodes, 
  int                     g_num_trie_nodes,
  constant const char*    g_trie_edge_labels,
  constant const ushort*  g_trie_edge_targets,
  constant const char*    g_dice,
  constant const char*    g_cell_neighbors,
  global char*            g_best_boards,
  global ushort*          g_best_scores,
  int                     g_offset)
{
  int id = get_global_id(0);

  ulong seed = id + g_offset;
  char board[BOARD_SIZE];
  char best_board[BOARD_SIZE];
  ushort best_score = 0;
  
  for (int i = 0; i < MAX_EPOCH; i++) {
    make_random_board(board, g_dice, &seed);
    ushort score = eval_board(g_trie_nodes, g_num_trie_nodes, g_trie_edge_labels, g_trie_edge_targets,
                              g_dice, g_cell_neighbors, board);
    if (score > best_score) {
      best_score = score;
      for (int j = 0; j < BOARD_SIZE; j++) best_board[j] = board[j];
    }
  }
  g_best_scores[id] = best_score;
  for (int j = 0; j < BOARD_SIZE; j++) g_best_boards[j + id*BOARD_SIZE] = best_board[j];
}