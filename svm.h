#include<vector>
#include<iostream>
#include"mpi.h"
#include "omp.h"

//typedef double double;

using namespace std;

class Timer{
public:
    void Start(){
        start = MPI_Wtime();
    }
    void End(){
        end = MPI_Wtime();
        sum += end-start;
    }
    double start,end,sum=0;
    double all;
};

struct TimeProfile {
   static Timer checkSV,cmGather,saveModel;
   static Timer train;
   static Timer cpKernel;
   static Timer cpShrinking;
   static Timer cmDeltaA;
   static Timer cmLoadBalance;
   static Timer cpSelect,cpSelect1,cpSample,cpAlpha,cpGradient,cmSelect,cmSelect1,cmSample;
   static Timer readAll,cmScatter;
};

/**
* @brief 样本数据
*/
struct SvmFeature {
    double value=0; //样本属性值
    int id=0;   //样本属性id
};

/**
* @brief 样本数据
*/
struct SvmSample {
    int index;  //索引
    int label;  //标签
    double twoNorm; //二范数
    vector<SvmFeature> features;    //属性数组
};

/**
* @brief 传输样本信息
*/
struct TranInfo{
        double alpha_;  //样本所对应的alpha
        double xSquare_;   //样本数据平方
        double G_;
        int y_; //样本标签
        int globalIndex_;
};

struct activeInfo{
        int active;
        int rank;
};

/**
* @brief 核函数类型：线性核函数，多项式核函数，高斯核函数
*/
enum { LINEAR, POLY, RBF};

/**
* @brief 支持向量机参数
*/
struct SvmParameter {
	int kernelType; //核函数类型
    int maxIter;    //最大迭代次数
    int commIter;   //通信迭代次数（每commIter次迭代后通信一次）
	double gamma;	//poly/rbf核函数参数(超参数)
    double coef;    //poly核函数参数
    int degree; //poly核函数参数
	double cacheSize; //kernel cache 大小（MB）
	double epsilon; //停机条件
	double hyperParmC;	//超参数C
	int shrinking;	/* use the shrinking heuristics */
    int loadbalance;
};

struct SupportVector{
    int boundSV;
    vector<SvmSample> SVs;
    vector<double> alpha;
};

struct svm_problem{
    int numAllSamples;  //总样本个数
    int numLocalSamples;  //节点样本个数
    //int numLocalSamples;   //节点中样本个数
    int numAllPos;
    int numAllNeg;
    int numLocalPos;
    int numLocalNeg;
    int localStart;

    //int elements;
    int numValues;
    int numLocalValues;
    int numProc;
    int * sendValues;
    int * sendSamples;
    int * v_displs;
    int * s_displs;

    SvmFeature *features;
    double* value;    //属性值
    int* labels;
    int* id;    //属性id
    int* index; //数据行索引
    double* alpha;

    SvmFeature *allFeatures;
    double* allValue; 
    int* allLabels; 
    int* allId; 
    int* allIndex;
    int maxId;
};

class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(int index, double **data, int len, bool *cacheStatus);
	void swap_index(int i, int j, int *globalindex);
    int getInfo(int index);
    long int getSize();
    double getData(int index);
    double *getTranData(int start, int tranSize);
    double *getTran();
    void setTranData(int start, int tranSize);
    void resetShrunk(int *index, int num, bool *cacheStatus);
private:
	int l;
	long int size;  //cache中空闲的列数
    long int columnSize;    //cache中可以存放的总列数
    long int dataSize;  //loadbalance传输过程中可传输的double的个数
    int localL;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		double *data;
		int len;		// data[0,len) is cached in this entry
        int logicLen;
	};

    double *tran;
    double *newData;

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

class Model {
public:
    Model(){
        param.gamma = 1;
        param.hyperParmC = 1; 
        param.epsilon = 1e-3;
        param.kernelType = RBF;
        param.coef = 0;
        param.degree = 3;
        param.maxIter = 1000;
        param.commIter = 10;
        param.cacheSize = 100;
    }
    virtual ~Model(){}

    SvmParameter getParam(){return param;}
    int getSVnum(){return allSVSamples;}
    SvmFeature *getSVfeatures(){return allSVfeatures;}
    double *getSVyAlpha(){return SVyAlpha;}
    int *getSVindex(){return allSVindex;}
    double *getSVnorm(){return norm;}
    double getRho(){return rho;}

    // Uses alpha values to decide which samples are support vectors and stores
    // their information.
    //void CheckSupportVector(Solver& solver, const SvmParameter& p);
    void CheckSupportVector_mpi(int maxid, double *alpha, SvmFeature *features, int *labels, int *index, int numSamples, const SvmParameter& p, double rho);
    int svm_save_model(const char *model_file_name);
    int svm_load_model(const char *model_file_name);
    //void saveTimeInfo(char *filename);

private:
    SvmParameter param;
    double rho;

    int numAllPos;
    int numAllNeg;
    int numLocalPos;
    int numLocalNeg;

    SvmFeature *SVfeatures;
    int* SVlabels;
    int* SVindex; //数据行索引
    double *SValpha;

    SvmFeature *allSVfeatures;
    int* allSVlabels; 
    int* allSVindex;
    double *allSValpha;
    double *norm;

    int processId;
    double *SVyAlpha;
    int maxId;
    //int numSV;
    //int numSVvalue;
    //double* value;    //属性值
    //int* labels;
    //int* id;    //属性id
    //int* index; //数据行索引

    int allSVSamples;  //总样本个数
    int SVSamples;  //节点样本个数

    //int elements;
    int allSVvalues;
    int SVvalues;
    int numProc;
    int * recValues;
    int * recSamples;
    int * v_displs;
    int * s_displs;

    //Timer checkSV,cmGather,saveModel,syn;
};

class Solver{
public:
    Solver(const SvmParameter &param);
    //Solver(){}
    ~Solver(){}

    /**
    * @brief 读入训练数据到各个节点
    *
    * @param filename 训练数据文件名
    */
    void read_mpi(const char* filename);

    float calKernel(int i, int j);
    float calMaxKernel(int i, SvmFeature *j);
    float calMinKernel(int i, SvmFeature *j);
    float calKernelwithLabel(int i, int j);
    float calKernel(SvmFeature *i, SvmFeature *j);
    double innerProduct(int i, int j);
    double innerProduct(SvmFeature *i, SvmFeature *j);
    double innerProduct(int i, SvmFeature *j);

    double *getAlpha();

    /**
    * @brief svm求解器
    *
    * @param l 样本数量
    * @param alpha 拉格朗日变量
    * 
    */
    //void Solve(Kernel &q, const SvmParameter &param, SvmDataset &dataset, Model &model);//attention SvmDataset &data 常量？引用？
    void Solve_mpi(Model &model, const SvmParameter &param);
    void Solve2_mpi(Model &model, const SvmParameter &param);
    void Solve0_mpi(Model &model, const SvmParameter &param);
    //void saveTimeInfo(char *filename);

    void predict_mpi(Model &model);

    int getLocalSampleNum();
    int getAllSampleNum();
    int getIterNum();

protected:
    struct {
        double f;
        int rank;
    }fmin,fmax,gfmin,gfmax,*allFmin,*allFmax;
    int gfmaxI;

    TranInfo infoMax,infoMin;
    SvmFeature *sampleMax;
    SvmFeature *sampleMin;
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
    char *alpha_status;
    bool *cache_status;
    int *shrunk_status;
    //int l;
    //int *y;
    double *alpha;
    double *G;
    double obj;
    const double C;
    const double eps;
    int shrinking;
    int loadbalance;
    //Kernel *Q;
    //double *QD;
    
    int numAllSamples;  //总样本个数
    int numLocalSamples;  //节点样本个数
    //int numLocalSamples;   //节点中样本个数
    int numAllPos;
    int numAllNeg;
    int numLocalPos;
    int numLocalNeg;
    int localStart;

    //int elements;
    int numValues;
    int numLocalValues;
    int numProc;
    int * sendValues;
    int * sendSamples;
    int * v_displs;
    int * s_displs;
    
    int * recValues;
    int * recSamples;

    int *sampleIndex;   //所有样本的索引
    int *globalIndex;   //节点内部样本的全局索引

    SvmFeature *features;
    SvmFeature *features2;
    //double* value;    //属性值
    int* labels;
    //int* id;    //属性id
    int* index; //数据行索引
    int* index2; //数据行索引
    //double* alpha;
    int *predictLabels;
    int allCorrect;

    SvmFeature *allFeatures;
    //double* allValue; 
    int* allLabels; 
    //int* allId; 
    int* allIndex;
    int maxId;

    SvmFeature *SVfeatures;
    int* SVindex;
    double *SVyAlpha;
    double *xSquare2;

    double *xSquare;
    const int kernelType;
    const double gamma;
    const double coef;
    const int degree;
    double *QD;

    const double cacheSize;
    Cache *cache;

    int activeSize; //活动集大小，即未被shrinking的样本个数
    activeInfo  activeSize_;    //带有节点信息的活动集大小
    int minActive,maxActive;    //所有节点中活动集大小的最大最小值，负载均衡问题
    int *activeNum; //数组大小为commSz，存储各节点活动集大小信息
    activeInfo *activeNum_;
    int avgActive;

     const int commIter;
    // double *localAlpha;
    // double *globalAlpha;
    // int *localIndex;
    // int *globalIndex;
    // SvmSample *localData;
    // SvmSample *globalData;

    // double ub,lb,Gub,Glb;
    double rho;

    int access=0;
    int hitLen=0;
    int hitPar=0;
    int iter;

    //master节点中聚集的要进行负载均衡的数据信息
    int ATSamples;	//allTranSamples
	int ATValues;
	int *ATLabels;
	double *ATSquares;
	SvmFeature *ATFeatures;
	double *ATAlpha;
	double *ATG;
	int *ATGlobalIndex;

    //被shrinking且存储在cache中的节点全局索引
    int *shrunk_in_cache;
    int *Allshrunk_in_cache;
    int AllSC;

    /**
    * @brief 选择更新变量
    *
    * @param i 要更新变量的索引
    */
    int selectWorkingSet(int &out_i);
    int selectWorkingSet2(int &out_j, double &obj_min, double *Qmax);

    int select_working_set(int &out_i, int &out_j);

    int select_working_set2(int &out_i, int &out_j);

    void update_shrunk(double Gmax, double Gmin);
    void do_shrinking(int &work_i, int &work_j);
    bool be_shrunk(int i, double Gmax, double Gmin);
    bool be_shrunk(int i);
    void swap_index(int i, int j);

    void load_balance();
    void load_balance2(int &work_i);

    void update_alpha_status(int i)
	{
		if(alpha[i] >= C)
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
    double calculate_rho();
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }

    double *get_Qmax(int i, int len) {
		access++;
		double *data;
		int start, j;
        start = cache->get_data(i,&data,len,cache_status);
		if(start < len){
			if(start!=0)
				hitPar++;
//#pragma omp parallel for private(j) schedule(guided)
			for(j=start;j<len;j++){
				//data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
                data[j] = calMaxKernel(j,sampleMax);
                //cout<<"num threads: "<<omp_get_num_threads()<<endl;
            }
		}
		else
			hitLen++;
		return data;
	}

    double *get_Qmin(int i, int len) {
		access++;
		double *data;
		int start, j;
		start = cache->get_data(i,&data,len,cache_status);
		if(start < len){
			if(start!=0)
				hitPar++;
//#pragma omp parallel for private(j) schedule(guided)
			for(j=start;j<len;j++){
				//data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
                data[j] = calMinKernel(j,sampleMin);
                //cout<<"num threads: "<<omp_get_num_threads()<<endl;
            }
		}
		else
			hitLen++;
		return data;
	}

};

/**
* @brief 从命令行中读入参数
*
* @param argc 命令行字符串个数
* @param argv 命令行字符串
* @param dataFileName 数据文件名
* @param modelFileName 模型文件名
*/
void parseCommandLine(int argc, char **argv, SvmParameter *param, 
    char *dataFileName, char *modelFileName, char *timeFileName);

/**
* @brief 参数检查
*
* @param dataset 数据集
* @param param 支持向量机参数
*/
void checkParameter( const SvmParameter &param);

void exit_with_help(int i);
void saveTimeInfo(char *filename, SvmParameter param, Solver &solve);

void quick_sort(activeInfo *s, int l, int r);
