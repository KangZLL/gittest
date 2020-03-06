#include <iostream>
#include <fstream>
//#include <string.h>

#include"svm.h"

struct svm_problem prob;

void exitwithHelp(){

}

int main(int argc, char **argv){
    char trainDataFile[256];
    char modelFile[256];
    char timeFile[256];

    MPI_Init(&argc, &argv);
    int myrank;
    int commSz;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    //cout<<"num process   max thread: "<<commSz<<" "<<omp_get_max_threads()<<endl;
#pragma omp parallel
{
    // if(omp_get_thread_num()==0)
    // cout<<"num threads: "<<omp_get_num_threads()<<endl;
    //if(myrank == 0)
    //cout<<"num threads: "<<omp_get_thread_num()<<endl;
}
    //SvmDataset dataSet;
    SvmParameter param;
    //SupportVector SV;
    Model model;

    // if(!parseCommandLine(argc, argv, &param, trainDataFile, modelFile)){
    //     cerr << "Error in function: parseCommandLine()" << endl;
    //     exit(1);
    // }
    
	// if(!dataSet.read_mpi(trainDataFile)){
    //     cerr << "Error in function: read_mpi()" << endl;
    //     exit(1);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // double start = MPI_Wtime();

    parseCommandLine(argc, argv, &param, trainDataFile, modelFile, timeFile);
	checkParameter(param);
    
    Solver S(param);
    S.read_mpi(trainDataFile);

    S.Solve2_mpi(model,param);
    //S.saveTimeInfo(timeFile);
    model.svm_save_model(modelFile);
    //model.saveTimeInfo(timeFile);

    saveTimeInfo(timeFile,param,S);

    // MPI_Barrier(MPI_COMM_WORLD);
    // double end = MPI_Wtime();
    // if(myrank == 0){
    //     ofstream outfile(timeFile, ios::out | ios::app);
    //     cout << endl<< "____________________________________" << endl;
	// 	outfile << endl<< "____________________________________" << endl;
    //     cout<<"all time: "<<end-start<<endl;
    //     outfile<<"all time: "<<end-start<<endl;
    //     outfile.close();
    // }
     /* IMPORTANT */
    //cout<<"#"<<endl;
//if(myRank==0)
	
    //model.CheckSupportVector(S, param);
    // int myrank;
    // int df;
    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // //cout<<myrank<<"#"<<endl;
    // MPI_Allreduce(&myrank,&df,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    //cout<<df;
    MPI_Finalize();

    return 0;
}