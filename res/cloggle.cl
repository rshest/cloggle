#define GENE_POOL_SIZE 10
#define MAX_NEIGHBORS 8
#define MAX_WORD_LEN 16
#define BOARD_SIDE 5
#define DIE_FACES 6
#define BOARD_SIZE (BOARD_SIDE*BOARD_SIDE)
#define MAX_TRIE_SIZE 1100

#define NUM_MUTATE_TYPES      10

#define MUTATE_SWAP_NEIGHBORS 0
#define MUTATE_ROLL_FACE      1
#define MUTATE_ROLL_FACE2     2

#define MUTATE_SWAP_RANDOM    3
#define MUTATE_SWAP_RANDOM3   4
#define MUTATE_SWAP_RANDOM4   5
#define MUTATE_ROLL_RANDOM    6
#define MUTATE_ROLL_RANDOM2   7
#define MUTATE_ROLL_RANDOM3   8
#define MUTATE_SWAP_ROLL      9


#define MAX_PLATEAU_AGE       200

//  trie node
typedef struct
{
  uchar   score;          //  node score, assuming terminal if non-0
  uchar   num_edges;      //  number of outgoing edges
  ushort  edges_offset;   //  offset in the edge label/target arrays
} Node;

//  poor-man random number generator (stolen from Java library implementation)
ulong rnd(ulong* seed) {
  return *seed = ((*seed) * 0x5DEECE66DL + 0xBL)&(((ulong)1 << 48) - 1);
}

//  shuffle array in-place (Fischer-Yites)
void shuffle_board(uchar* board, ulong* seed) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    int swap_with = rnd(seed)%(BOARD_SIZE - i);
    char tmp = board[swap_with];
    board[swap_with] = board[i];
    board[i] = tmp;
  }
}

//  creates a random board from the set of dice
void make_random_board(uchar* board, constant const uchar* g_num_dice, ulong* seed) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    board[i] = (unsigned char)(i*DIE_FACES + rnd(seed)%g_num_dice[i]);
  }
  shuffle_board(board, seed);
}

void swap(unsigned char* arr, int i, int j) {
  unsigned char tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}

uchar die_offs(uchar die) {
  return (die/DIE_FACES)*DIE_FACES;
}

void random_flip(uchar* board, int pos, constant const uchar* g_num_dice, ulong* seed) {
  uchar offs = board[pos]/DIE_FACES;
  int d = rnd(seed) % g_num_dice[offs];
  board[pos] = d + offs*DIE_FACES;
}


//  evaluates the board score
ushort eval_board(
  constant const Node*    g_trie_nodes, 
  int                     g_num_trie_nodes,
  constant const char*    g_trie_edge_labels,
  constant const ushort*  g_trie_edge_targets,
  constant const char*    g_dice,
  constant const char*    g_cell_neighbors,
  const uchar*            board
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
        char c = g_dice[board[cell]];
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
  global uchar*           g_boards,
  global ushort*          g_scores,
  int                     g_num_boards)
{
  uchar board[BOARD_SIZE];
  for (int i = 0; i < g_num_boards; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      board[j] = g_boards[i*BOARD_SIZE + j];
    }

    g_scores[i] = eval_board(g_trie_nodes, g_num_trie_nodes, g_trie_edge_labels, g_trie_edge_targets,
      g_dice, g_cell_neighbors, board);
  }
}

//  grinding kernel 
kernel void grind(
  constant const Node*    g_trie_nodes, 
  int                     g_num_trie_nodes,
  constant const char*    g_trie_edge_labels,
  constant const ushort*  g_trie_edge_targets,
  constant const char*    g_dice,
  constant const uchar*   g_num_dice,
  constant const char*    g_cell_neighbors,
  global uchar*           g_boards,
  global ushort*          g_scores,
  global ushort*          g_ages)
{
  int id = get_global_id(0);
  uchar board[BOARD_SIZE];
  ushort best_score = g_scores[id];
  ulong seed = id*7 + best_score*(g_ages[id] + 1);

  if (g_ages[id] >= MAX_PLATEAU_AGE) {
    //  first iteration, or score plateaued, init fresh
    make_random_board(board, g_num_dice, &seed);
    for (int j = 0; j < BOARD_SIZE; j++) g_boards[id*BOARD_SIZE + j] = board[j];
    g_ages[id] = 0;
    best_score = 0;
  } 
  
  int mutateType = rnd(&seed) % NUM_MUTATE_TYPES;
  int pivot_cell = rnd(&seed) % BOARD_SIZE;
  int pivot_cell2 = rnd(&seed) % BOARD_SIZE;

  const int MUTATE_STEPS[] = {  MAX_NEIGHBORS, DIE_FACES, DIE_FACES*DIE_FACES, 
    BOARD_SIZE, BOARD_SIZE, BOARD_SIZE,
    BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE };
  int nsteps = MUTATE_STEPS[mutateType];

  for (int i = 0; i < nsteps; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      board[j] = g_boards[id*BOARD_SIZE + j];
    }

    switch (mutateType) {
    case MUTATE_SWAP_RANDOM: {
      swap(board, i, pivot_cell);
    } break;
    case MUTATE_SWAP_RANDOM3: {
      int c1 = rnd(&seed) % BOARD_SIZE;
      int c2 = rnd(&seed) % BOARD_SIZE;
      int c3 = rnd(&seed) % BOARD_SIZE;
      swap(board, c1, c2);
      swap(board, c2, c3);
      swap(board, c3, c1);
    } break;
    case MUTATE_SWAP_RANDOM4: {
      int c1 = rnd(&seed) % BOARD_SIZE;
      int c2 = rnd(&seed) % BOARD_SIZE;
      int c3 = rnd(&seed) % BOARD_SIZE;
      int c4 = rnd(&seed) % BOARD_SIZE;
      swap(board, c1, c2);
      swap(board, c2, c3);
      swap(board, c3, c4);
      swap(board, c4, c1);
    } break;
    case MUTATE_SWAP_NEIGHBORS: {
      int neighbor = g_cell_neighbors[i + pivot_cell*(MAX_NEIGHBORS + 1)];
      if (neighbor >= 0) {
        swap(board, neighbor, pivot_cell);
      }
    } break;
    case MUTATE_ROLL_FACE: {
      board[pivot_cell] = i + die_offs(board[pivot_cell]);
    } break;
    case MUTATE_ROLL_FACE2: {
      int d1 = i % DIE_FACES; 
      int d2 = i / DIE_FACES;
      board[pivot_cell] =  d1 + die_offs(board[pivot_cell]);
      board[pivot_cell2] = d2 + die_offs(board[pivot_cell2]);
    } break;

    case MUTATE_ROLL_RANDOM: {
      int c1 = rnd(&seed) % BOARD_SIZE;
      random_flip(board, c1, g_num_dice, &seed);
    } break;
    case MUTATE_ROLL_RANDOM2: {
      int c1 = rnd(&seed) % BOARD_SIZE;
      int c2 = rnd(&seed) % BOARD_SIZE;
      random_flip(board, c1, g_num_dice, &seed);
      random_flip(board, c2, g_num_dice, &seed);
    } break;
    case MUTATE_ROLL_RANDOM3: {
      int c1 = rnd(&seed) % BOARD_SIZE;
      int c2 = rnd(&seed) % BOARD_SIZE;
      int c3 = rnd(&seed) % BOARD_SIZE;
      random_flip(board, c1, g_num_dice, &seed);
      random_flip(board, c2, g_num_dice, &seed);
      random_flip(board, c3, g_num_dice, &seed);
    } break;
    case MUTATE_SWAP_ROLL: {
      swap(board, i, pivot_cell);
      random_flip(board, i, g_num_dice, &seed);
      random_flip(board, pivot_cell, g_num_dice, &seed);
    } break;
    default: {}
    }

    ushort score = eval_board(g_trie_nodes, g_num_trie_nodes, g_trie_edge_labels, g_trie_edge_targets,
      g_dice, g_cell_neighbors, board);
    if (score > best_score) {
      best_score = score;
      for (int j = 0; j < BOARD_SIZE; j++) g_boards[id*BOARD_SIZE + j] = board[j];
    }
  }
  
  g_ages[id] += (g_scores[id] == best_score);
  g_scores[id] = best_score;

}