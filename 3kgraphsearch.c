/******************************************************************************
 * FILE: 3kgraphsearch.c
 * DESCRIPTION:  
 *   MPI Find 3k Graph for 32 vertex with lowest Dmin, Amin - C Version
 *   AUTHOR: Adam Pollack
 ******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

const int nnodes = 32; // Set before running

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */



// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Code that performs a parallel rank
// Holds the communicator rank of a process along with the corresponding number.
// This struct is used for sorting the values and keeping the owning process information
// intact.
typedef struct {
  int comm_rank;
  union {
    float f;
    int i;
  } number;
} CommRankNumber;

// Gathers numbers for TMPI_Rank to process zero. Allocates enough space given the MPI datatype and
// returns a void * buffer to process 0. It returns NULL to all other processes.
void *gather_numbers_to_root(void *number, MPI_Datatype datatype, MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  
  // Allocate an array on the root process of a size depending on the MPI datatype being used.
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);
  void *gathered_numbers;
  if (comm_rank == 0) {
    gathered_numbers = malloc(datatype_size * comm_size);
  }
  
  // Gather all of the numbers on the root process
  MPI_Gather(number, 1, datatype, gathered_numbers, 1, datatype, 0, comm);
  
  return gathered_numbers;
}

// A comparison function for sorting float CommRankNumber values
int compare_float_comm_rank_number(const void *a, const void *b) {
  CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
  CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
  if (comm_rank_number_a->number.f < comm_rank_number_b->number.f) {
    return -1;
  } else if (comm_rank_number_a->number.f > comm_rank_number_b->number.f) {
    return 1;
  } else {
    return 0;
  }
}

// A comparison function for sorting int CommRankNumber values
int compare_int_comm_rank_number(const void *a, const void *b) {
  CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
  CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
  if (comm_rank_number_a->number.i < comm_rank_number_b->number.i) {
    return -1;
  } else if (comm_rank_number_a->number.i > comm_rank_number_b->number.i) {
    return 1;
  } else {
    return 0;
  }
}

// This function sorts the gathered numbers on the root process and returns an array of
// ordered by the process's rank in its communicator. Note - this function is only
// executed on the root process.
int *get_ranks(void *gathered_numbers, int gathered_number_count, MPI_Datatype datatype) {
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);
  
  // Convert the gathered number array to an array of CommRankNumbers. This allows us to
  // sort the numbers and also keep the information of the processes that own the numbers
  // intact.
  CommRankNumber *comm_rank_numbers = malloc(gathered_number_count * sizeof(CommRankNumber));
  int i;
  for (i = 0; i < gathered_number_count; i++) {
    comm_rank_numbers[i].comm_rank = i;
    memcpy(&(comm_rank_numbers[i].number), gathered_numbers + (i * datatype_size), datatype_size);
  }
  
  // Sort the comm rank numbers based on the datatype
  if (datatype == MPI_FLOAT) {
    qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_float_comm_rank_number);
  } else {
    qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_int_comm_rank_number);
  }
  
  // Now that the comm_rank_numbers are sorted, create an array of rank values for each process. The ith
  // element of this array contains the rank value for the number sent by process i.
  int *ranks = (int *)malloc(sizeof(int) * gathered_number_count);
  for (i = 0; i < gathered_number_count; i++) {
    ranks[comm_rank_numbers[i].comm_rank] = i;
  }
  
  // Clean up and return the rank array
  free(comm_rank_numbers);
  return ranks;
}

// Gets the rank of the recv_data, which is of type datatype. The rank is returned
// in send_data and is of type datatype.
int TMPI_Rank(void *send_data, void *recv_data, MPI_Datatype datatype, MPI_Comm comm) {
  // Check base cases first - Only support MPI_INT and MPI_FLOAT for this function.
  if (datatype != MPI_INT && datatype != MPI_FLOAT) {
    return MPI_ERR_TYPE;
  }
  
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  
  // To calculate the rank, we must gather the numbers to one process, sort the numbers, and then
  // scatter the resulting rank values. Start by gathering the numbers on process 0 of comm.
  void *gathered_numbers = gather_numbers_to_root(send_data, datatype, comm);
  
  // Get the ranks of each process
  int *ranks = NULL;
  if (comm_rank == 0) {
    ranks = get_ranks(gathered_numbers, comm_size, datatype);
  }
  
  // Scatter the rank results
  MPI_Scatter(ranks, 1, MPI_INT, recv_data, 1, MPI_INT, 0, comm);
  
  // Do clean up
  if (comm_rank == 0) {
    free(gathered_numbers);
    free(ranks);
  }
}

int calcD(int dist[nnodes][nnodes]){
  int d = 0;
  int i,j;
  
  for(i = 0; i < nnodes; i++){
    for(j = i + 1; j < nnodes; j++){
      if(dist[i][j] > d){
        d = dist[i][j];
      }
    }
  }
  return d;
}

double calcA(int dist[nnodes][nnodes]){
  int i, j;
  double a = 0;
  
  for(i = 0; i < nnodes; i++){
    for(j = i + 1; j < nnodes; j++){
      a += dist[i][j];
    }
  }
  return a/((nnodes*(nnodes-1)/2));
}

void floyd(int dist[nnodes][nnodes])
{
  int i,j,k;
  
  for (k = 0; k < nnodes; ++k) 
  {
    for (i = 0; i < nnodes; ++i)
    {
      for (j = 0; j < nnodes; ++j)
      {
        if ((dist[i][k] * dist[k][j] != 0) && (i != j))
        {
          if ((dist[i][k] + dist[k][j] < dist[i][j]) || (dist[i][j] == 0))
          {
            dist[i][j] = dist[i][k] + dist[k][j];
          }
        }
      }
    }                   
  }
  
}

int randint(int n) {
  int divisor = RAND_MAX/(n);
  int retval;
  
  do{
    retval = rand() / divisor;
  }while (retval > n-1);
  return retval;
}

void adddegree(int *array, int nnodes){
  int check;
  int randIndex;
  int temp;
  int temp1;
  int j, k;
  int nodesavail;
  int lastIndex;
  
  
  randIndex = randint(nnodes-3) + 2;
  temp = array[0];
  temp1 = array[randIndex];
  
  for(j = 1; j < nnodes; j++){
    array[j-1] = array[j];
  }
  
  array[nnodes-1] = temp;
  array[randIndex-1] = array[nnodes-2];
  array[nnodes-2] = temp1;
  
  
  for(j = 1; j < (nnodes-2)/2; j++){
    nodesavail = nnodes-2*j;
    lastIndex = nodesavail-1;
    randIndex = randint(lastIndex) + 1;
    temp = array[0];
    temp1 = array[randIndex];
    
    check = abs(temp-temp1);
    
    if(check == 1){
      array[randIndex] = array[lastIndex];
      array[lastIndex] = temp1;
      randIndex = randint(lastIndex -1) + 1;
      temp1 = array[randIndex];
    }
    
    check = abs(temp-temp1);
    
    if(check == 1){
      array[randIndex] = array[lastIndex];
      array[lastIndex] = temp1;
      randIndex = randint(lastIndex -1) + 1;
      temp1 = array[randIndex];
    }
    
    for(k = 1; k < nodesavail; k++){
      array[k-1] = array[k];
    }
    
    array[lastIndex] = temp;
    array[randIndex-1] = array[lastIndex-1];
    array[lastIndex-1] = temp1;
    
    
  }
}

int main (int argc, char *argv[])
{
  int	numtasks,              /* number of tasks in partition */
taskid,                /* a task identifier */


adjs,                  /* # adjacency matrices for worker to create */
num_adjs,              /*#adjacencies overall  */
aveadj, extra, /* used to determine adj. matrices for each worker */
i,j, k;          /* misc */
int v[nnodes];
  
double starttime, endtime; /*Calculating runtime*/
MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);


srand((unsigned int)time(NULL));

num_adjs = 28000000; /*Total number of adjacency matrices. Can adjust */

aveadj = num_adjs/numtasks;
extra = num_adjs%numtasks;
  
adjs = (taskid <= extra) ? aveadj + 1 : aveadj; /*Assigns the number of adjacency matrices this node will work through*/
  
int twokadj[nnodes][nnodes]; /*Initialize to 0*/
memset(twokadj, 0, nnodes*nnodes*sizeof(int));
int threekadj[nnodes][nnodes];
int dist[nnodes][nnodes];
memset(dist, 0, nnodes*nnodes*sizeof(int));
int threekadjbest[nnodes][nnodes];
memset(threekadj,0,nnodes*nnodes*sizeof(int));
int diam;
double avg;
memset(threekadjbest,0,nnodes*sizeof(int));
int diambest = 100;
double avgbest = 100.0;
 
/*Set the index neighbors to 1 for the 2k 32 vertex graph */
for(i = 1; i < nnodes - 1; i++){ 
    twokadj[i][i+1] = 1;
    twokadj[i+1][i] = 1;
    twokadj[i][i-1] = 1;
    twokadj[i-1][i] = 1;
}
twokadj[0][nnodes-1] = 1;
twokadj[nnodes-1][0] = 1;
  
  /* Loop through # of adjacency matrices
   * In each loop, randomly generate the 3rd degrees
   * Calculating the distance matrix
   * Calculating Dmin and Amin
   * Storing the adj. matrix for Dmin and Amin if it beats global Dmin and Amin
   * In other words, if Dmin and Amin of some graph is <= to the one stored, replace
   * Report the adjacency matrix with lowest Dmin and Amin
   */
  
  starttime = MPI_Wtime(); /*Start timing*/
  
  for(i = 0; i < adjs; i++){
    //Reset threekadj
    for(j = 0; j < nnodes; j++){
      for(k = 0; k < nnodes; k++){
        threekadj[j][k] = twokadj[j][k];
      }
    }
    
    // Reset the vertex list
    int sum = 0;
    for(j = 0; j < nnodes; j++){
      v[j] = j;
      sum += v[j];
    }
    
    int sum_check;
    do{
      adddegree(v, nnodes); 
      k = 0; //keep track of whether or not the vertex list is appropriate
      sum_check = 0;
      for(j = 0; j < nnodes; j+=2){
        if(abs(v[j] - v[j+1]) == 1)
          k = 1;
        sum_check += (v[j] + v[j+1]);
      }
    } while ((k == 1) || (sum_check) != (nnodes*(nnodes-1)/2));
    
    //With the updated list, add appropriate adjacencies
    for(j = 0; j < nnodes; j+=2){
      threekadj[v[j]][v[j+1]] = 1;
      threekadj[v[j+1]][v[j]] = 1;
    }
    
    
    for(j = 0; j < nnodes; j++){
      for(k = 0; k < nnodes; k++){
        dist[j][k] = 0;
      }
    }
    
    for(j = 0; j < nnodes; j++){
      for(k = 0; k < nnodes; k++){
        dist[j][k] = threekadj[j][k];
      }
    }
    
    floyd(dist);
    
    diam = calcD(dist);
    avg = calcA(dist);
    
    if(diam<=diambest && avg<=avgbest){
      diambest = diam;
      avgbest = avg;
      for(j = 0; j < nnodes; j++){
        for(k = 0; k < nnodes; k++){
          threekadjbest[j][k] = threekadj[j][k];
        }
      }
      
    }
  }//end of loop for finding best adjacency matrix
  
  int rank;
  TMPI_Rank(&avgbest, &rank, MPI_FLOAT, MPI_COMM_WORLD);
  endtime = MPI_Wtime();
  if(rank == 0){
    printf("Printing Best Adjacency Matrix of Run:\n");
    for (i = 0; i < nnodes; i++)
    {
      for (j = 0; j < nnodes; j++)
      {
        printf ("%4d", threekadjbest[i][j]);
      }
      printf("\n");
    }
    printf("Diameter: %d\nAverage: %.6f\n", diambest, avgbest);
    printf("Searching %d graphs on %d processors took %.6f seconds\n", num_adjs, numtasks, endtime-starttime);
    
  }
 

MPI_Finalize();
}

