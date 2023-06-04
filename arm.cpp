#include<iostream>
#include<iomanip>
#include<semaphore.h>
#include <sys/time.h>
#include "mpi.h"
#include<unistd.h>
#include<cstring>
#include<omp.h>
using namespace std;
const int N=1024;
const int p=1;
float** m;
//float** n;




void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] = m[k][j] + m[i][j];
}


void normal()//
{
    struct timeval starttime,endtime;
    double timeuse;
    gettimeofday(&starttime,NULL);
    m_reset();
    
    for (int k = 0; k < N; k++)
        {
                for (int j = k + 1; j < N; j++)
                {
                        m[k][j] = m[k][j] / m[k][k];

                }
                m[k][k] = 1.0;
                for (int i = k + 1; i < N; i++)
                {
                        for (int j = k + 1; j < N; j++)
                        {
                                m[i][j] -= m[i][k] * m[k][j];
                        }
                        m[i][k] = 0;
                }

        }
    gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout<<"time:"<<timeuse<<endl;
}

void mpi_block() {

    
    struct timeval starttime,endtime;
    double timeuse;
    gettimeofday(&starttime,NULL);
    m_reset();
    
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int r1, r2;
    int num = N / numprocs;
    r1 = myid * num;
    if(myid == numprocs - 1) {
		r2 = (myid + 1) * N / numprocs;
	} else {
		r2 = (myid + 1) * num;
	}

    for(int k = 0; k < N; k++ )
    {
        if(k >= r1 && k < r2)
            {
            for(int j = k + 1; j < N; j++)
            {
				m[k][j] = m[k][j] / m[k][k];
			}
            m[k][k] = 1;
            for(int j = 0; j < numprocs; j++)
            {
                if(j != myid)
                    MPI_Send(&m[k][0], N, MPI_FLOAT, j, k, MPI_COMM_WORLD);
            }
        } else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
        for(int i=r1; i<r2; i++)
            {
            if(i == k)
            {
                continue;
            }
            for(int j = k + 1; j < N; j++)
            {
				m[i][j] = m[i][j]-m[k][j]*m[i][k];
			}
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout<<"block_time:"<<timeuse<<endl;
}

void mpi_pipeline() {

    
    m_reset();
    struct timeval starttime,endtime;
    double timeuse;
    gettimeofday(&starttime,NULL);
 
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Request request;
    MPI_Status status;

    int up = myid - 1;
    int down = myid + 1;
	if (myid == 0) {
        up = numprocs - 1;
    }
	if (myid == numprocs - 1) {
        down = 0;
    }
    int num = N / numprocs;

    int p = 0;
    for(int k = 0; k < N; k++ ) {
        int source = k % numprocs;
        if(myid == source) {
            for(int j = k + 1; j < N; j++) {
				m[k][j] = m[k][j] / m[k][k];
			}
            m[k][k] = 1;
            MPI_Isend(&m[p][0], N, MPI_FLOAT, down, k, MPI_COMM_WORLD, &request);
            p++;
        } else {
			MPI_Irecv(&m[p][0], N, MPI_FLOAT, up, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            int picked = status.MPI_TAG;
			if (down != source) {
                MPI_Isend(&m[p][0], N, MPI_FLOAT, down, picked, MPI_COMM_WORLD, &request);
            }
		}
        for (int i = p; i < num; ++i) {
            for(int j = k+1; j < N; j++) {
                m[i][j] = m[i][j] - m[k][j]*m[i][k];
            }
            m[i][k] = 0;
		}
    }

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&endtime,NULL);
    timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
    timeuse/=1000000;/*转换成秒输出*/
    cout<<"pipeline_time:"<<timeuse<<endl;

}

void mpi_cycle()
{
    
    m_reset();
    struct timeval starttime,endtime;
    double timeuse;
    gettimeofday(&starttime,NULL);
	int comm_sz;
	int my_rank;

	MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	int r1=my_rank*(N-N%comm_sz)/comm_sz;
	int r2;
	if(my_rank!=comm_sz-1)
		r2=my_rank*(N-N%comm_sz)/comm_sz+(N-N%comm_sz)/comm_sz-1;
	else r2=N-1;

	if(my_rank==0){
		m_reset();
		for(int i=1;i<comm_sz;i++)
			MPI_Send(m,N*N,MPI_FLOAT,i,0,MPI_COMM_WORLD);
	}
	else MPI_Recv(m,N*N,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	for(int k=0;k<N;k++)
	{
		if(r1<=k&&k<=r2){
			for(int j=k+1;j<N;j++)
				m[k][j]=m[k][j]/m[k][k];
			m[k][k]=1.0;
			for(int num=my_rank+1;num<comm_sz;num++)
				MPI_Send(&m[k][0],N,MPI_FLOAT,num,1,MPI_COMM_WORLD);
		}
		else {
			if(k<=r2)
			MPI_Recv(&m[k][0],N,MPI_FLOAT,k/((N-N%comm_sz)/comm_sz),1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//	break;
		}


		int i;
		if((r1<=k+1)&&(k+1<=r2))
			i=k+1;
		if(k+1<r1)i=r1;
		if(k+1>r2)i=N;
		for(i;i<=r2;i++)
		{
			for(int j=k+1;j<N;j++)
				m[i][j]=m[i][j]-m[i][k]*m[k][j];
			m[i][k]=0;
		}
	}

	if(my_rank!=0)
                MPI_Send(&m[r1][0],N*(r2-r1+1),MPI_FLOAT,0,2,MPI_COMM_WORLD);
	else for(int q=1;q<comm_sz;q++){
                        if(q!=comm_sz-1)
                        	MPI_Recv(&m[q*(N-N%comm_sz)/comm_sz][0],N*(r2-r1+1),MPI_FLOAT,q,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        else MPI_Recv(&m[q*(N-N%comm_sz)/comm_sz][0],N*(r2-r1+1+N%comm_sz),MPI_FLOAT,q,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }


	gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout<<"cycle_time:"<<timeuse<<endl;
}


void print()
{
    for(int i=0;i<20;i++)
    {
        for(int j=0;j<20;j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
}


int main(int argc, char *argv[])
{

    int provide = 7;
    MPI_Init_thread (&argc, &argv,MPI_THREAD_SERIALIZED,&provide);
    m = new float* [N];
	for (int i = 0; i < N; i++)
    {
		m[i] = new float[N];
	}

    //normal();

    //mpi_block();
    //mpi_pipeline();
    mpi_cycle();
    //mpi_simd();
    //mpi_omp();
    //mpi_omp_simd();
    print();
    MPI_Finalize();
    //print();
	return 0;
}
