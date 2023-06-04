#include<iostream>
#include<iomanip>
#include<semaphore.h>
#include "mpi.h"
#include<unistd.h>
#include<cstring>
#include<omp.h>
#include <windows.h>
using namespace std;
const int N=2560;
const int p=1;
float** m;
//float** n;
LARGE_INTEGER freq, t1, t2;



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
    double sumtime=0;
    QueryPerformanceFrequency(&freq);
    for(int x = 0 ;x < p ; x++ )
    {
    m_reset();
    QueryPerformanceCounter(&t1);
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
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
     }
    cout << "normal_time: " << sumtime/p << "ms" << endl;
}

void mpi_block() {

    double sumtime=0;
    QueryPerformanceFrequency(&freq);
    for(int x = 0 ;x < p ; x++ )
    {
    m_reset();
    QueryPerformanceCounter(&t1);
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
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    MPI_Barrier(MPI_COMM_WORLD);

     }
    cout << "normal_time: " << sumtime/p << "ms" << endl;
}

void mpi_pipeline() {

    double sumtime=0;
    QueryPerformanceFrequency(&freq);

    m_reset();
    QueryPerformanceCounter(&t1);
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
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "pipeline_time: " << sumtime << "ms" << endl;

}

void mpi_cycle()
{
    double sumtime=0;
    QueryPerformanceFrequency(&freq);

    m_reset();
    QueryPerformanceCounter(&t1);
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


	QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "cycle_time: " << sumtime << "ms" << endl;
}

void mpi_simd() {

    double sumtime=0;
    QueryPerformanceFrequency(&freq);

    m_reset();
    QueryPerformanceCounter(&t1);
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

    for(int k = 0; k < N; k++ ) {
        if(k >= r1 && k < r2) {
            __m128 vt = _mm_set1_ps(m[k][k]);
            for(int j = k + 1; j < N; j+=4) {
				__m128 va = _mm_loadu_ps(&m[k][j]);
			    va = _mm_div_ps(va, vt);
			    _mm_storeu_ps(&m[k][j], va);
			    if (j + 8 > N) {//处理末尾
				    while (j < N) {
					    m[k][j] /= m[k][k];
					    j++;
				    }
				    break;
			    }
			}
            m[k][k] = 1;
            for(int j = 0; j < numprocs; j++) {
                if(j != myid) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, j, k, MPI_COMM_WORLD);
                }
            }
        } else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
        for(int i = r1; i < r2; i++) {
            if(i == k) {
                continue;
            }
            __m128 vaik = _mm_loadu_ps(&m[i][k]);
            for(int j = k + 1; j + 4 < N; j+=4) {

                __m128 vakj = _mm_loadu_ps(&m[k][j]);
				__m128 vaij = _mm_loadu_ps(&m[i][j]);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(&m[i][j], vaij);
                if (j + 8 > N) {//处理末尾
					while (j < N) {
						m[i][j] -= m[i][k] * m[k][j];
						j++;
					}
					break;
				}
			}
            m[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "sse_time: " << sumtime << "ms" << endl;
}

void mpi_omp() {

    double sumtime=0;
    QueryPerformanceFrequency(&freq);

    m_reset();
    QueryPerformanceCounter(&t1);
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
     #pragma omp parallel
    for(int k = 0; k < N; k++ ) {
        if(k >= r1 && k < r2) {
            for(int j = k + 1; j < N; j++) {
				m[k][j] = m[k][j] / m[k][k];
			}
            m[k][k] = 1;
            for(int j = 0; j < numprocs; j++) {
                if(j != myid)
                    MPI_Send(&m[k][0], N, MPI_FLOAT, j, k, MPI_COMM_WORLD);
            }
        }
        else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
        #pragma omp for
        for(int i = r1; i < r2; i++) {

            for(int j = k + 1; j < N; j++) {
                if(i <= k) continue;
				m[i][j] = m[i][j]-m[k][j]*m[i][k];
			}
            m[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "sse_time: " << sumtime << "ms" << endl;
}

void mpi_omp_simd() {

    double sumtime=0;
    QueryPerformanceFrequency(&freq);

    m_reset();
    QueryPerformanceCounter(&t1);
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
	#pragma omp parallel
    for(int k = 0; k < N; k++ ) {
        if(k >= r1 && k < r2) {
            __m128 vt = _mm_set1_ps(m[k][k]);
            for(int j = k + 1; j < N; j+=4) {
				__m128 va = _mm_loadu_ps(&m[k][j]);
			    va = _mm_div_ps(va, vt);
			    _mm_storeu_ps(&m[k][j], va);
			    if (j + 8 > N) {//处理末尾
				    while (j < N) {
					    m[k][j] /= m[k][k];
					    j++;
				    }
				    break;
			    }
			}
            m[k][k] = 1;
            for(int j = 0; j < numprocs; j++) {
                if(j != myid) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, j, k, MPI_COMM_WORLD);
                }
            }
        } else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
		#pragma omp for
        for(int i = r1; i < r2; i++) {
            if(i == k) {
                continue;
            }
            __m128 vaik = _mm_loadu_ps(&m[i][k]);
            for(int j = k + 1; j + 4 < N; j+=4) {

                __m128 vakj = _mm_loadu_ps(&m[k][j]);
				__m128 vaij = _mm_loadu_ps(&m[i][j]);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(&m[i][j], vaij);
                if (j + 8 > N) {//处理末尾
					while (j < N) {
						m[i][j] -= m[i][k] * m[k][j];
						j++;
					}
					break;
				}
			}
            m[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "sse_time: " << sumtime << "ms" << endl;
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
