#include <iostream>
#include<string.h>
//#include <string.h>

#include"svm.h"

void predict_mpi(){}

int main(int argc, char **argv){
    char testDataFile[256];
    char modelFile[256];
    MPI_Init(&argc, &argv);
    int i = 1;
    if(i < argc)    
        strcpy(testDataFile, argv[i]);
    else
        exit_with_help(1);
        
    //模型文件名
    if(++i < argc)  
        strcpy(modelFile, argv[i]);
    else
        exit_with_help(1);

    // strcpy(testDataFile, argv[1]);
    // strcpy(modelFile, argv[2]);

    //SvmParameter param;
    //SupportVector SV;
    Model model;
    model.svm_load_model(modelFile);
    Solver S(model.getParam());
    S.read_mpi(testDataFile);
    S.predict_mpi(model);
    MPI_Finalize();

    //S.Solve(model,param);

    return 0;
}