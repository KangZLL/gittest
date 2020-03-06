#include <iostream>
#include <iomanip>
#include <string.h>
#include<sstream>
#include <fstream>
#include<string>
#include<math.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#ifdef WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif
//#include <cstdlib>
#include"svm.h"

#ifndef INT_MAX
#define INT_MAX       2147483647
#endif

Timer TimeProfile::checkSV;
Timer TimeProfile::cmGather;
Timer TimeProfile::saveModel;
Timer TimeProfile::train;
Timer TimeProfile::cpSelect;
Timer TimeProfile::cpSelect1;
Timer TimeProfile::cpSample;
Timer TimeProfile::cpShrinking;
Timer TimeProfile::cpAlpha;
Timer TimeProfile::cpGradient;
Timer TimeProfile::cmSelect;
Timer TimeProfile::cmSelect1;
Timer TimeProfile::cmSample;
Timer TimeProfile::readAll;
Timer TimeProfile::cmScatter;
Timer TimeProfile::cpKernel;
Timer TimeProfile::cmDeltaA;
Timer TimeProfile::cmLoadBalance;

// #ifndef min
// template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
// #endif
// #ifndef max
// template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
// #endif
template <class T> static inline void Swap(T& x, T& y) { T t=x; x=y; y=t; }
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help(int i)
{	
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if(myrank==0){
		if(i==0){
			printf(
			"Usage: svm_train [options] training_set_file [model_file]\n"
			"options:\n"
			"-k kernel_type : set type of kernel function (default 2)\n"
			"	0 -- linear: u'*v\n"
			"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
			"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
			"-d degree : set degree in kernel function (default 3)\n"
			"-g gamma : set gamma in kernel function (default 1)\n"
			"-r coef0 : set coef0 in kernel function (default 0)\n"
			"-c hyperParmC : set the parameter C(default 1)\n"
			"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
			"-m cacheSize : set cache memory size in MB (default 100)\n"
			"-s shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
			"-i iteration : num of iteration between shrinking (default 1000)\n"
			);
			MPI_Abort(MPI_COMM_WORLD,99);
			//exit(1);
		}
		else if(i==1){
			printf(	"Usage: svm_predict test_set_file model_file\n"	);
			MPI_Abort(MPI_COMM_WORLD,99);
			//exit(1);
		}
	}
}

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	int commSz;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	localL = l/commSz;
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	//cout<<sizeof(head_t)<<endl;
	//for(int i=0;i<l;i+=100){
	//	cout<<i<<" "<<&head[i]<<" "<<&head[i]-head<<" "<<(&head[i]-head)/sizeof(head_t)<<endl;
	//}
	size /= sizeof(double);
	size -= l * sizeof(head_t) / sizeof(double);
	dataSize = size*0.2;
	tran = new double[dataSize];
	//size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	size = max((long int)10, size/localL);	//***size为列数而不再是double值的个数
	columnSize = size;
	//size = 20000;
	lru_head.next = lru_head.prev = &lru_head;
	if(rank == 0){
		if((float)size/l<=1)
			std::cout<<"Percentage of cache in total Q: "<<(float)size/l*100<<"%"<<std::endl;
		//if((float)size/(l*l/commSz)<=1)
			//std::cout<<"Percentage of cache in total Q: "<<(float)size/(l*l/commSz)*100<<"%"<<std::endl;
		else 
			std::cout<<"Percentage of cache in total Q: "<<100<<"%"<<std::endl;
	}
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
	delete[] tran;
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::getInfo(int index){
	return head[index].len;
}

double Cache::getData(int index){
	return head[index].len>0?head[index].data[0]:-1;
}

long int Cache::getSize(){
	return columnSize - size;
}

double *Cache::getTran(){
	return tran;
}

void Cache::resetShrunk(int *index, int num, bool *cacheStatus){
	for(int i=0;i<num;i++){
		head_t *h = &head[index[i]];
		int shrunkIndex = h-head;
		if(shrunkIndex!=index[i])
			cout<<"resetShrunk wrong"<<endl;
		cacheStatus[index[i]] = 0;
		lru_delete(h);
		free(h->data);
		//size += old->len;
		size++;
		h->data = 0;
		h->len = 0;
	}
}

double *Cache::getTranData(int start, int tranSize){
	long int column=0;
	//newData = NULL;
	if(tranSize*columnSize>dataSize){
		dataSize = tranSize*columnSize;
		tran = (double *)realloc(tran,sizeof(double)*dataSize);
	}
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next){
		if(h->len>0){
			for(int i = start; i < start + tranSize; i++){
				tran[column*tranSize + i - start] = h->data[i];
			}
			// newData = (double *)malloc(newData,sizeof(double)*(start));
			// memcpy(newData,h->data,start);
			// free(h->data);
			// h->data = newData;
			// h->len = start;
			// newData = NULL;
			column++;
		}
	}
	if(column != columnSize - size){
		cout<<"cache size column is wrong1"<<endl;
		cout<<column<<" "<<columnSize<<" "<<size<<endl;
	}
	return tran;
}

void Cache::setTranData(int start, int tranSize){
	long int column=0;
	//newData = NULL;
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next){
		if(h->len>0){
			if(h->len < start + tranSize){
				h->data = (double *)realloc(h->data,sizeof(double)*(start + tranSize));
				h->len = start + tranSize;
			}
			// else if(h->len > start + tranSize){
			// 	newData = (double *)malloc(sizeof(double)*(start + tranSize));
			// 	memcpy(newData,h->data,start + tranSize);
			// 	free(h->data);
			// 	h->data = newData;
			// 	h->len = start + tranSize;
			// 	newData = NULL;
			// }
			for(int i = start; i < start + tranSize; i++){
				//tran[column*size+i-start]=h->data[i];
				h->data[i] = tran[column*tranSize + i - start];

			}
			column++;
		}
	}
	if(column != columnSize - size){
		cout<<"cache size column is wrong1"<<endl;
		cout<<column<<" "<<columnSize<<" "<<size<<endl;
	}
}

int Cache::get_data(int index, double **data, int len, bool *cacheStatus){	

	// int myRank,commSz;
	// MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    // MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	// if(h->logicLen!=h->len && len >h->logicLen){
	// 	int startIndex = h->logicLen;
	// 	h->logicLen=h->len;
	// 	return startIndex;
	// }

	if(more > 0){
		if(more != len)
			cout<<"bingo!~!!"<<endl;
		// free old space
		//while(size < more)
		if(size==0){
		//while(size < more){
			//cout<<"!";
			head_t *old = lru_head.next;
			int oldindex = old-head;
			cacheStatus[oldindex] = 0;
			lru_delete(old);
			free(old->data);
			//size += old->len;
			size++;
			old->data = 0;
			old->len = 0;
			//old->logicLen = 0;
		}

		// allocate new space
		h->data = (double *)realloc(h->data,sizeof(double)*len);
		//size -= more;
		size--;
		//h->logicLen = len;
		Swap(h->len,len);
	}
	else if(more < 0){

	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j, int *globalindex){
	if(i==j) return;
	// int globalI = globalindex[i];
	// int globalJ = globalindex[j];

	// if(head[globalI].len) lru_delete(&head[globalI]);
	// if(head[globalJ].len) lru_delete(&head[globalJ]);
	// Swap(head[globalI].data,head[globalJ].data);
	// Swap(head[globalI].len,head[globalJ].len);
	// if(head[globalI].len) lru_insert(&head[globalI]);
	// if(head[globalJ].len) lru_insert(&head[globalJ]);

	if(i>j) Swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next){
		if(h->len > i){
			if(h->len > j){
				Swap(h->data[i],h->data[j]);
				//h->logicLen--;
			}
			else{
				cout<<"sdfsdfa"<<endl;
				// give up
				lru_delete(h);
				free(h->data);
				//size += h->len;	//******注意size的含义变了
				size++;
				h->data = 0;
				h->len = 0;
				//h->logicLen = 0;
			}
		}
	}
}

float Solver::calKernel(SvmFeature *i, SvmFeature *j){
	switch (kernelType){
        case LINEAR:
            return innerProduct(i,j);
        case POLY:
            return powi(gamma*innerProduct(i,j) + coef,degree);
        case RBF:
            //cout<<"in:"<<innerProduct(data[i], data[j])<<endl;
            //cout<<"form"<<exp(-gamma * (data[i].twoNorm - 2 * innerProduct(data[i], data[j]) + data[j].twoNorm))<<endl;
            //cout<<"form"<<infoMax.xSquare_<<" "<<innerProduct(i, j)<<" "<<infoMin.xSquare_<<endl;
			return exp(-gamma * (infoMax.xSquare_ - 2 * innerProduct(i, j) + infoMin.xSquare_));
        default:
            cerr << "Error: Unknown kernel type" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
            return 0.0;
    }
}
float Solver::calMaxKernel(int i, SvmFeature *j){
	switch (kernelType){
        case LINEAR:
            return innerProduct(i,j);
        case POLY:
            return powi(gamma*innerProduct(i,j) + coef,degree);
        case RBF:
            //cout<<"in:"<<innerProduct(data[i], data[j])<<endl;
            //cout<<"form"<<infoMax.xSquare_<<" "<<innerProduct(i, j)<<" "<<infoMin.xSquare_<<endl;
            return exp(-gamma * (xSquare[i] - 2 * innerProduct(i, j) + infoMax.xSquare_));
        default:
            cerr << "Error: Unknown kernel type" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
            return 0.0;
    }
}
float Solver::calMinKernel(int i, SvmFeature *j){
	switch (kernelType){
        case LINEAR:
            return innerProduct(i,j);
        case POLY:
            return powi(gamma*innerProduct(i,j) + coef,degree);
        case RBF:
            //cout<<"in:"<<innerProduct(data[i], data[j])<<endl;
            //cout<<"form"<<infoMax.xSquare_<<" "<<innerProduct(i, j)<<" "<<infoMin.xSquare_<<endl;
            return exp(-gamma * (xSquare[i] - 2 * innerProduct(i, j) + infoMin.xSquare_));
        default:
            cerr << "Error: Unknown kernel type" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
            return 0.0;
    }
}
double Solver::innerProduct(SvmFeature *i, SvmFeature *j){
    double norm = 0;
    int m = 0;
    int n = 0;
    while(i[m].id!=0 && j[n].id!=0){
		//cout<<"dot:"<<i[m].id<<" "<<i[m].value<<endl;
        if(i[m].id==j[n].id){
        //if(id[m]==id[n]){
            norm += i[m].value*j[n].value;
            //norm += value[m]*value[n];
            m++;
            n++;
        }
        else if(i[m].id>j[n].id){
        //else if(id[m]>id[n]){
            n++;
        }
        else
            m++;
    }
    return norm;
}

double Solver::innerProduct(int i, SvmFeature *j){
    double norm = 0;
    int m = index[i];
    int n = 0;
    while(features[m].id!=0 && j[n].id!=0){
        if(features[m].id==j[n].id){
        //if(id[m]==id[n]){
            norm += features[m].value*j[n].value;
            //norm += value[m]*value[n];
            m++;
            n++;
        }
        else if(features[m].id>j[n].id){
        //else if(id[m]>id[n]){
            n++;
        }
        else
            m++;
    }
    return norm;
}

float Solver::calKernel(int i, int j){

    switch (kernelType){
        case LINEAR:
            return innerProduct(i,j);
        case POLY:
            return powi(gamma*innerProduct(i,j) + coef,degree);
        case RBF:
            //cout<<"in:"<<innerProduct(data[i], data[j])<<endl;
            //cout<<"form"<<exp(-gamma * (data[i].twoNorm - 2 * innerProduct(data[i], data[j]) + data[j].twoNorm))<<endl;
            //cout<<"form"<<infoMax.xSquare_<<" "<<innerProduct(i, j)<<" "<<infoMin.xSquare_<<endl;
			//cout<<gamma<<endl;
			return exp(-gamma * (xSquare[i] - 2 * innerProduct(i, j) + xSquare2[j]));
        default:
            cerr << "Error: Unknown kernel type" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
            return 0.0;
    }
}

float Solver::calKernelwithLabel(int i, int j){

    switch (kernelType){
        case LINEAR:
            return (i == j)?xSquare[i]:innerProduct(i,j)*labels[i]*labels[j];
        case POLY:
            return powi(gamma*innerProduct(i,j) + coef,degree)*labels[i]*labels[j];
        case RBF:
            //cout<<"in:"<<innerProduct(data[i], data[j])<<endl;
            //cout<<"form"<<exp(-gamma * (data[i].twoNorm - 2 * innerProduct(data[i], data[j]) + data[j].twoNorm))<<endl;
            return (i == j)?1:exp(-gamma * (xSquare[i] - 2 * innerProduct(i, j) + xSquare[j]))*labels[i]*labels[j];
        default:
            cerr << "Error: Unknown kernel type" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
            return 0.0;
    }
}

double Solver::innerProduct(int i, int j){
    double norm = 0;
    int m = index[i];
    int n = index2[j];
    //while(m<index[i+1] && n<index[j+1]){
	while(features[m].id!=0 && features2[n].id!=0){
        if(features[m].id==features2[n].id){
        //if(id[m]==id[n]){
            norm += features[m].value*features2[n].value;
            //norm += value[m]*value[n];
            m++;
            n++;
        }
        else if(features[m].id>features2[n].id){
        //else if(id[m]>id[n]){
            n++;
        }
        else
            m++;
    }
    return norm;
}

Solver::Solver(const SvmParameter &param):C(param.hyperParmC),eps(param.epsilon),commIter(param.commIter),kernelType(param.kernelType),
            gamma(param.gamma),degree(param.degree),coef(param.coef),cacheSize(param.cacheSize),shrinking(param.shrinking),loadbalance(param.loadbalance){}

void Solver::read_mpi(const char* filename){
    int myRank, commSz;
    numAllSamples = 0;
    numLocalSamples = 0;
    localStart = -1;
    numAllPos = 0;
    numAllNeg = 0;
    numLocalNeg = 0;
    numLocalPos = 0;
	numValues = 0;
	numLocalValues = 0;
    int label;

    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    numProc = commSz;
	//MPI_Barrier(MPI_COMM_WORLD);

    if(myRank == 0){
		TimeProfile::readAll.Start();
		cout << "------------------------------------" << endl;
		cout<<"The parameters of svm:	C:"<<C<<" eps:"<<eps<<" gamma:"<<gamma<<endl;
        cout << "------------------------------------" << endl;
		cout<<"Reading all data..."<<endl;
        ifstream infile(filename);
        if(!infile.is_open()){
            cerr << "Error: Fail to open " << filename << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
        }

		maxId = 0;
        int instMaxId;
        string str;
        //获取样本数据个数和element
        while(getline(infile, str)){
			instMaxId = 0;
            numAllSamples++;
            stringstream strStream(str);
            string strData;
			strStream>>strData;
			//cout<<strData<<endl;
            //getline(strStream, strData, ' ');
            //while(getline(strStream, strData, ' ')){
			while(strStream>>strData){
				//cout<<strData<<" t"<<endl;
				instMaxId++;
                numValues++;
            }
			if(instMaxId > maxId){
				maxId = instMaxId;
			}
			//cout<<numAllSamples<<" "<<numValues<<endl;
        //system("pause");
		}
		if(commSz>numAllSamples){
			cerr << "Error: The num of proc is larger than samples!" << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
            //exit(1);
		}

        infile.clear();                                                                                                                                              
        infile.seekg(0, ios::beg);
		//cout<<numAllSamples<<" "<<numValues<<endl;
		//system("pause");
        allLabels = new int[numAllSamples];
		allIndex = new int[numAllSamples+1];
        allFeatures = new SvmFeature[numAllSamples*(maxId+1)];//numValues+numAllSamples];
		sampleIndex = new int[numAllSamples];

        //maxId = 0;
        //int instMaxId;
        int j = 0;
        int i;
        for(i=0;i<numAllSamples;i++){
			j = 0;
			sampleIndex[i]=i;
            //instMaxId = 0;
            getline(infile, str);
            //allIndex[i] = j;  //样本数据对于属性的索引
			allIndex[i] = i*(maxId+1);

            stringstream strStream(str);
            string strData;
            //getline(strStream, strData, ' ');
			strStream>>strData;
            label = atoi(strData.c_str());
            if(label == 1)
                numAllPos++;
            else if(label == -1)//||label == 2)
                numAllNeg++;
            else{
                cerr << "Error: Wrong label in line:" << i + 1 << endl;
				MPI_Abort(MPI_COMM_WORLD,99);
                //exit(1);
            }
            allLabels[i] = label;
            //cout<<myRank<<":"<<allLabels[i]<<"# "<<i<<" ";
            //while(getline(strStream, strData, ' ')){
			while(strStream>>strData){
                stringstream strStream2(strData);

                getline(strStream2, strData, ':');
                //allFeatures[j].id = atoi(strData.c_str());
				allFeatures[i*(maxId+1)+j].id = atoi(strData.c_str());

                //instMaxId = allFeatures[j].id;
				//instMaxId++;
				if(j>0 && allFeatures[i*(maxId+1)+j].id<=allFeatures[i*(maxId+1)+j-1].id){
					cerr << "Error: The feature id in line "<<i+1<<" is wrong!" << endl;
					MPI_Abort(MPI_COMM_WORLD,99);
				}
				//cout<<"instMaxId"<<instMaxId<<endl;
                getline(strStream2, strData);
                //allFeatures[j].value = atof(strData.c_str());
				allFeatures[i*(maxId+1)+j].value = atof(strData.c_str());
                //cout<<allFeatures[j].id<<":"<<allFeatures[j].value<<"# ";

                j++;
            }
			// allFeatures[j].id = 0;
			// allFeatures[j].value = 0;
			allFeatures[i*(maxId+1)+j].id = 0;
			allFeatures[i*(maxId+1)+j].value = 0;
			//cout<<allFeatures[j].id<<":"<<allFeatures[j].value<<"# ";
			j++;
			//allFeatures[j].id = 0;
            //cout<<endl;
            //if(instMaxId > maxId){
				//maxId = instMaxId;
			//}
        }
		//cout<<"maxid"<<maxId<<endl;
        //allIndex[i]=numValues+numAllSamples;
		allIndex[i]=i*(maxId+1);	
		//cout<<i<<" "<<allIndex[i]<<" "<<numValues<<" "<<numAllSamples;
		infile.close();
		TimeProfile::readAll.End();
		cout<<"done"<<endl;
    }
	//MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::cmScatter.Start();
    MPI_Bcast(&numAllSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&maxId, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//cout<<"maxid"<<maxId<<endl;
    numLocalSamples = numAllSamples/numProc;
    
    //cout<<numLocalSamples<<endl;

    if(myRank==0){
        sendSamples = new int[numProc];
        s_displs = new int[numProc];
        int numPer = numAllSamples/numProc;
        int remainder = numAllSamples%numProc;

		sendValues = new int[numProc];
		v_displs = new int[numProc];
		for(int i=0;i<numProc;i++){
            if(i<remainder)
                sendSamples[i] = numPer + 1;
            else
                sendSamples[i] = numPer;
			
			if(i==0){	
				v_displs[i] = 0;
                s_displs[i] = 0;
			}
		   	else{ 
				v_displs[i] = v_displs[i-1]+sendValues[i-1];
                s_displs[i] = s_displs[i-1]+sendSamples[i-1];
			}
            sendValues[i] = allIndex[s_displs[i]+sendSamples[i]] - allIndex[s_displs[i]];
			//cout<<s_displs[i]+sendSamples[i]<<" "<<allIndex[s_displs[i]+sendSamples[i]]<<" "<<allIndex[s_displs[i]]<<endl;;
		}
	}
	//cout<<sendValues[0]<<endl;
    MPI_Scatter(sendValues, 1, MPI_INT, &numLocalValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sendSamples, 1, MPI_INT, &numLocalSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    features = new SvmFeature[numLocalValues];
	features2 = features;
	index = new int[numLocalSamples+1]; 
	index2 = index;
    labels = new int[numLocalSamples];

	globalIndex = new int[numLocalSamples];
	cache_status = new bool[numAllSamples]();

    MPI_Scatterv(allFeatures, sendValues, v_displs, MPI_DOUBLE_INT, features, numLocalValues, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(allIndex, sendSamples, s_displs, MPI_INT, index, numLocalSamples, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(allLabels, sendSamples, s_displs, MPI_INT, labels, numLocalSamples, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(sampleIndex, sendSamples, s_displs, MPI_INT, globalIndex, numLocalSamples, MPI_INT, 0, MPI_COMM_WORLD);

	for(int i=1;i<numLocalSamples;i++){
		index[i] = (index[i] - index[0]);
	}
	index[0] = 0;
	//cout<<numLocalSamples<<" "<<numLocalValues<<endl;
	index[numLocalSamples] = numLocalValues;

	TimeProfile::cmScatter.End();
	//MPI_Barrier(MPI_COMM_WORLD);
  		
  	if(myRank==0){
		cout<<"Num of all data:"<<numAllSamples<<endl;
		cout<<"The num of data in per proc:"<<numLocalSamples<<endl;
	  	delete[] sendValues;
	  	delete[] v_displs;
        delete[] sendSamples;
	  	delete[] s_displs;
	}

    cache = new Cache(numAllSamples,(long int)(cacheSize*(1<<20)));

    xSquare = new double[numLocalSamples];
	xSquare2 = xSquare;
    QD = new double[numLocalSamples];
    for(int i = 0; i < numLocalSamples; i++){
        xSquare[i] = innerProduct(i,i);
        QD[i] = calKernel(i,i);
    }
}

double *Solver::getAlpha(){
    return alpha;
}

int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<numLocalSamples;t++){
    //cout<<"y["<<t<<"]: "<<y[t]<<endl;
		if(labels[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}
    }
	int i = Gmax_idx;
	// const double *Q_i = NULL;
	// if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
	// 	Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<numLocalSamples;j++)
	{
		if(labels[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2){
					Gmax2 = G[j];
					//Gmin_idx = j;
				}
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*calKernel(i,j);
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2){
					Gmax2 = -G[j];
					//Gmin_idx = j;
				}
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*calKernel(i,j);
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}
    //int myrank;
    //MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    //std::cout<<"Gmax_idx Gmin_idx: "<<Gmax_idx<<" "<<Gmin_idx<<std::endl;
    //cout<<"rank Gmax: "<<myrank<<" "<<Gmax<<" "<<Gmax2<<" "<<alpha[Gmax_idx]<<" "<<alpha[Gmin_idx]<<endl;
	if(Gmax+Gmax2 < eps || Gmin_idx == -1)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

int Solver::select_working_set2(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmin = INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	//for(int t=0;t<numLocalSamples;t++){
	int t;
//#pragma omp parallel for //private(t) schedule(guided)
    for(t=0;t<activeSize;t++){
		if(labels[t]==+1){
			//if(!is_upper_bound(t))
			if(alpha[t] != C)
				if(-G[t] > Gmax){
					Gmax = -G[t];
					Gmax_idx = t;
				}
            //if(!is_lower_bound(t))
			if(alpha[t] != 0)
				if(-G[t] < Gmin){
					Gmin = -G[t];
					Gmin_idx = t;
				}
		}
		else{
			//if(!is_lower_bound(t))
			if(alpha[t] != 0)
				if(G[t] > Gmax){
					Gmax = G[t];
					Gmax_idx = t;
				}
            //if(!is_upper_bound(t))
			if(alpha[t] != C)
				if(G[t] < Gmin){
					Gmin = G[t];
					Gmin_idx = t;
				}
		}
    }
	
    // int myrank;
    // MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    // std::cout<<"Gmax_idx Gmin_idx: "<<Gmax_idx<<" "<<Gmin_idx<<std::endl;
    // cout<<"rank Gmax: "<<myrank<<" "<<Gmax<<" "<<Gmax2<<" "<<alpha[Gmax_idx]<<" "<<alpha[Gmin_idx]<<endl;
	// if(Gmax+Gmax2 < eps || Gmin_idx == -1)
	// 	return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

int Solver::selectWorkingSet(int &out_i){
    double Gmax = -INF;
	//double Gmax2 = -INF;
	int Gmax_idx = -1;
	//int Gmin_idx = -1;
	//double obj_diff_min = INF;

	for( int j=0 ; j<numLocalSamples;  j++){
		if(labels[j]==+1 && !is_upper_bound(j)){
			if(-G[j] >= Gmax){
				Gmax = -G[j];
				Gmax_idx = j;
			}
		}
		else if(labels[j]==-1 && !is_lower_bound(j)){
			if(G[j] >= Gmax){
				Gmax = G[j];
				Gmax_idx = j;
			}
		}
	}
    //int myrank;
    //MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    //cout<<"rank Gmax: "<<myrank<<" "<<G[Gmax_idx]<<" "<<alpha[Gmax_idx]<<" "<<Gmax_idx<<endl;
	//(*value) = Gmax;
	//if ( Gmax <= eps )
		//return 1;
	out_i = Gmax_idx;
	return 0;
}

int Solver::selectWorkingSet2(int &out_j, double &obj_min, double *Qmax){
    //double Gmax = -INF;
	double Gmax2 = -INF;
	//int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	//double *Q_max = get_Qmax(infoMax.globalIndex_,numLocalSamples);

	//for(int j=0;j<numLocalSamples;j++){
		int j;
//#pragma omp parallel for //private(j) schedule(guided)
	for(j=0;j<activeSize;j++){
		if(labels[j]==+1){
			//if (!is_lower_bound(j)){
			if(alpha[j] != 0){
				//double grad_diff=gfmax.f+G[j];
				double grad_diff=-infoMax.G_*infoMax.y_+G[j];
				if (G[j] >= Gmax2){
					Gmax2 = G[j];
					//Gmin_idx = j;
				}
				if (grad_diff > 0){
					double obj_diff;
					//double quad_coef = 2-2.0*calMaxKernel(j,sampleMax);
					double quad_coef = 2-2.0*Qmax[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff < obj_diff_min){
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else{
			//if (!is_upper_bound(j)){
			if(alpha[j] != C){
				//double grad_diff= gfmax.f-G[j];
				double grad_diff=-infoMax.G_*infoMax.y_-G[j];
				if (-G[j] >= Gmax2){
					Gmax2 = -G[j];
					//Gmin_idx = j;
				}
				if (grad_diff > 0){
					double obj_diff;
					//double quad_coef = 2-2.0*calMaxKernel(j,sampleMax);
					double quad_coef = 2-2.0*Qmax[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff < obj_diff_min){
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	out_j = Gmin_idx;
	obj_min = obj_diff_min;
	return 0;
}

void Solver::swap_index(int i, int j){

	//cache->swap_index(i,j);
	cache->swap_index(i, j, globalIndex);

	Swap(QD[i],QD[j]);
	//Swap(cache_status[globalIndex[i]],cache_status[globalIndex[j]]);
	Swap(globalIndex[i],globalIndex[j]);
	Swap(xSquare[i],xSquare[j]);
	//Swap(xSquare2[i],xSquare2[j]);
	Swap(labels[i],labels[j]);
	//Swap(index[i],index[j]);
	memcpy(&features[index[i]],&features[index[j]],sizeof(SvmFeature)*(maxId+1));
	//Swap(index2[i],index2[j]);
	Swap(G[i],G[j]);
	Swap(shrunk_status[i],shrunk_status[j]);
	Swap(alpha_status[i],alpha_status[j]);
	Swap(alpha[i],alpha[j]);
	//swap(p[i],p[j]);
	//swap(active_set[i],active_set[j]);
	//swap(G_bar[i],G_bar[j]);
}

bool Solver::be_shrunk(int i){
	//if(is_upper_bound(i))
	// if(alpha[i] == C)
	// {
	// 	if(labels[i]==+1)
	// 		return(-G[i] > Gmax);
	// 	else
	// 		return(G[i] < Gmin);
	// }
	//else if(is_lower_bound(i))
	if(alpha[i] == 0)
	{
		if(labels[i]==+1)
			return(-G[i] < gfmin.f);
		else
			return(G[i] > gfmax.f);
	}
	else
		return(false);
}

bool Solver::be_shrunk(int i, double Gmax, double Gmin){
	//if(is_upper_bound(i))
	// if(alpha[i] == C)
	// {
	// 	if(labels[i]==+1)
	// 		return(-G[i] > Gmax);
	// 	else
	// 		return(G[i] < Gmin);
	// }
	//else if(is_lower_bound(i))
	if(alpha[i] == 0)
	{
		if(labels[i]==+1)
			return(-G[i] < Gmin);
		else
			return(G[i] > Gmax);
	}
	else
		return(false);
}

void Solver::do_shrinking(int &work_i, int &work_j){
	int i;
	//double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	//double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	int numSC=0;
	for(i=0;i<activeSize;i++){
		if(shrinking == 1){
			if (shrunk_status[i]>100){
				activeSize--;
				while (activeSize > i){
					if (shrunk_status[activeSize]<=100){
						if(work_i == activeSize){
							work_i = i;
						}
						if(work_j == activeSize){
							work_j = i;
						}
						swap_index(i,activeSize);
						break;
					}
					activeSize--;
				}
			}
		}
		else if(shrinking == 2){
			//cout<<"asdf"<<endl;
			//if (be_shrunk(i, gfmax.f, gfmin.f)){
			if (be_shrunk(i)){
				activeSize--;
				if(work_i == i){
					cout<<"wring"<<endl;
				}
				if(work_j == i){
					cout<<"wring"<<endl;
				}
				if(cache_status[globalIndex[i]] == 1){
					shrunk_in_cache[numSC] = globalIndex[i];
					numSC++;
				}
				while (activeSize > i){
					//if (!be_shrunk(activeSize, gfmax.f, gfmin.f)){
					if (!be_shrunk(activeSize)){
						if(work_i == activeSize){
							work_i = i;
						}
						if(work_j == activeSize){
							work_j = i;
						}
						swap_index(i,activeSize);
						break;
					}
					else if(cache_status[globalIndex[activeSize]] == 1){
						shrunk_in_cache[numSC] = globalIndex[activeSize];
						numSC++;
					}
					activeSize--;
				}
			}
		}
	}

	//cout<<"shrunk cache: "<<numSC<<endl;

	// int commSz;
    // int myRank;
    // MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    // MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// MPI_Allreduce(&numSC,&AllSC,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	// // //MPI_Reduce(&tranSamples,&ATSamples,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	// if(myRank == 0)
	// 	cout<<"shrunk cache: "<<AllSC<<endl;
	// MPI_Allgather(&numSC, 1, MPI_INT, recSamples, 1, MPI_INT, MPI_COMM_WORLD);
    // //MPI_Gather(&tranValues, 1, MPI_INT, recValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	// for(int k=0;k<commSz;k++){
	// 	if(k==0){	
	// 		//v_displs[k] = 0;
	// 		s_displs[k] = 0;
	// 	}
	// 	else{
	// 		//v_displs[k] = v_displs[k-1] + recValues[k-1];
	// 		s_displs[k] = s_displs[k-1] + recSamples[k-1];
	// 	}
	// }

	// //MPI_Allgather(&activeSize_, 1, MPI_2INT, activeNum_, 1, MPI_2INT, MPI_COMM_WORLD);
	// MPI_Allgatherv( shrunk_in_cache, numSC, MPI_INT, Allshrunk_in_cache, recSamples, s_displs, MPI_INT, MPI_COMM_WORLD);

	// cache->resetShrunk(Allshrunk_in_cache, AllSC, cache_status);
	// MPI_Gatherv( &features[minActive*(maxId+1)], tranValues, MPI_DOUBLE_INT, ATFeatures, recValues, v_displs, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
	// MPI_Gatherv( &labels[minActive], tranSamples, MPI_INT, ATLabels, recSamples, s_displs, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Gatherv( &globalIndex[minActive], tranSamples, MPI_INT, ATGlobalIndex, recSamples, s_displs, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Gatherv( &xSquare[minActive], tranSamples, MPI_DOUBLE, ATSquares, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// MPI_Gatherv( &alpha[minActive], tranSamples, MPI_DOUBLE, ATAlpha, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// MPI_Gatherv( &G[minActive], tranSamples, MPI_DOUBLE, ATG, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Solver::update_shrunk(double Gmax, double Gmin){

	for(int i=0;i<activeSize;i++){
		// if(alpha[i] == C){
		// 	if(labels[i]==+1 && -G[i] > Gmax)
		// 		shrunk_status[i]++;
		// 	else if(labels[i]==-1 && G[i] < Gmin)
		// 		shrunk_status[i]++;
		// 	else
		// 		shrunk_status[i] = 0;
		// }
		//else if(is_lower_bound(i))
		if(alpha[i] == 0){
			if(labels[i]==+1 && -G[i] < Gmin)
				shrunk_status[i]++;
			else if(labels[i]==-1 && G[i] > Gmax)
				shrunk_status[i]++;
			else
				shrunk_status[i] = 0;
		}
		else
			shrunk_status[i] = 0;
	}
}
void Solver::load_balance(){

	int commSz;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	
	int tranSamples;
	int tranValues;

	tranSamples = activeSize - minActive;
	tranValues = tranSamples*(maxId+1);

	if(myRank == 3)
		cout<<"1"<<endl;

	MPI_Reduce(&tranValues,&ATValues,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&tranSamples,&ATSamples,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	MPI_Gather(&tranSamples, 1, MPI_INT, recSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&tranValues, 1, MPI_INT, recValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(myRank == 3)
		cout<<"2"<<endl;
	if(myRank == 0){
		for(int k=0;k<commSz;k++){
			if(k==0){	
				v_displs[k] = 0;
				s_displs[k] = 0;
			}
			else{
				v_displs[k] = v_displs[k-1] + recValues[k-1];
				s_displs[k] = s_displs[k-1] + recSamples[k-1];
			}
		}
	}

	//MPI_Allgather(&activeSize_, 1, MPI_2INT, activeNum_, 1, MPI_2INT, MPI_COMM_WORLD);

	MPI_Gatherv( &features[minActive*(maxId+1)], tranValues, MPI_DOUBLE_INT, ATFeatures, recValues, v_displs, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( &labels[minActive], tranSamples, MPI_INT, ATLabels, recSamples, s_displs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( &globalIndex[minActive], tranSamples, MPI_INT, ATGlobalIndex, recSamples, s_displs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( &xSquare[minActive], tranSamples, MPI_DOUBLE, ATSquares, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( &alpha[minActive], tranSamples, MPI_DOUBLE, ATAlpha, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( &G[minActive], tranSamples, MPI_DOUBLE, ATG, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(myRank == 3)
		cout<<"3"<<endl;

	int numLocalTran = ATSamples/commSz;
    
    //cout<<numLocalSamples<<endl;

    if(myRank==0){
        //sendSamples = new int[commSz];
        //s_displs = new int[commSz];
        int numPer = ATSamples/commSz;
        int remainder = ATSamples%commSz;

		//sendValues = new int[commSz];
		//v_displs = new int[commSz];
		for(int i=0;i<commSz;i++){
            if(i<remainder)
                sendSamples[i] = numPer + 1;
            else
                sendSamples[i] = numPer;
			
			if(i==0){	
				v_displs[i] = 0;
                s_displs[i] = 0;
			}
		   	else{ 
				v_displs[i] = v_displs[i-1]+sendValues[i-1];
                s_displs[i] = s_displs[i-1]+sendSamples[i-1];
			}
			sendValues[i] = sendSamples[i]*(maxId+1);
            //sendValues[i] = allIndex[s_displs[i]+sendSamples[i]] - allIndex[s_displs[i]];
			//cout<<s_displs[i]+sendSamples[i]<<" "<<allIndex[s_displs[i]+sendSamples[i]]<<" "<<allIndex[s_displs[i]]<<endl;;
		}
	}

	MPI_Scatter(sendValues, 1, MPI_INT, &tranValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sendSamples, 1, MPI_INT, &tranSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
	activeSize = minActive + tranSamples;

    MPI_Scatterv(ATFeatures, sendValues, v_displs, MPI_DOUBLE_INT, &features[minActive*(maxId+1)], tranValues, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ATAlpha, sendSamples, s_displs, MPI_DOUBLE, &alpha[minActive], tranSamples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ATSquares, sendSamples, s_displs, MPI_DOUBLE, &xSquare[minActive], tranSamples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ATG, sendSamples, s_displs, MPI_DOUBLE, &G[minActive], tranSamples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ATLabels, sendSamples, s_displs, MPI_INT, &labels[minActive], tranSamples, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ATGlobalIndex, sendSamples, s_displs, MPI_INT, &globalIndex[minActive], tranSamples, MPI_INT, 0, MPI_COMM_WORLD);
  		
	//quick_sort(activeNum_,0,commSz);

	if(myRank == 3)
		cout<<"4"<<endl;
	
}

void Solver::load_balance2(int &work_i){
	int commSz;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	MPI_Status statusSend[7];
	MPI_Request requestSend[7];
	MPI_Status statusRecv[7];
	MPI_Request requestRecv[7];

	activeSize_.active = activeSize;
	//activeSize_.rank = myRank;
	activeSize_.rank = work_i*commSz+myRank;
	//将各个节点中的activesize汇总
	MPI_Allgather(&activeSize_, 1, MPI_2INT, activeNum_, 1, MPI_2INT, MPI_COMM_WORLD);

	//求出平均的activesize
	int sumActive=0;
	int activeRank;
	for(int i=0;i<commSz;i++){
		sumActive+=activeNum_[i].active;
		activeRank = activeNum_[i].rank % commSz;
		if(activeRank == gfmax.rank){
			gfmaxI = activeNum_[i].rank / commSz;
		}
		activeNum_[i].rank = activeRank;
	}
	int reminder = (sumActive%commSz==0)?0:1;
	avgActive = sumActive/commSz+reminder;	//注意：是平均值取进1
	if(myRank==0)
		cout<<"avg: "<<avgActive<<endl;

	//将activesize数组按升序排序
	quick_sort(activeNum_,0,commSz-1);
	

	if(myRank == 3)
		cout<<"load balance: "<<activeNum_[commSz-1].active-activeNum_[0].active<<" "<<activeNum_[0].active<<" "<<activeNum_[commSz-1].active<<endl;

	//int *gfmaxRank = new int[commSz];
	//vector<int> gfmaxRank(commSz,gfmax.rank);
	//大循环直到所有节点中活动集大小为平均大小后退出
	int LBrank = gfmax.rank;
	int LBi = gfmaxI;
	while(activeNum_[commSz-1].active >avgActive+5){
		int begin=0;
		int end=commSz-1;
		int a,b;
		//每次循环中分别取头和尾的一对节点进行load balance
		while(begin<end){
			if((a=avgActive-activeNum_[begin].active) > 0 && (b=activeNum_[end].active-avgActive) > 0){
				int tranNum = a>b?b:a;
				activeNum_[begin].active += tranNum;
				activeNum_[end].active -= tranNum;
				if(activeNum_[end].rank == gfmax.rank ){//&& work_i>=activeNum_[end].active){
					//int tran_i;
					//MPI_Recv(&tran_i,1,MPI_INT,activeNum_[end].rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					//当gfmax的索引i的数据属于要传输的数据时，更改gfmax.rank和索引i的信息
					if(gfmaxI >= activeNum_[end].active && gfmaxI < activeNum_[end].active + tranNum){
						LBrank = activeNum_[begin].rank;
						LBi = gfmaxI-activeNum_[end].active+activeSize;
					}
				}
				if(myRank == activeNum_[begin].rank){
					//解决负载平衡后gfmax.rank变化的问题
					// if(activeNum_[end].rank == gfmax.rank ){//&& work_i>=activeNum_[end].active){
					// 	//int tran_i;
					// 	//MPI_Recv(&tran_i,1,MPI_INT,activeNum_[end].rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					// 	//当gfmax的索引i的数据属于要传输的数据时，更改gfmax.rank和索引i的信息
					// 	if(gfmaxI >= activeNum_[end].active && gfmaxI < activeNum_[end].active + tranNum){
					// 		gfmax.rank = myRank;
					// 		work_i = gfmaxI-activeNum_[end].active+activeSize;
					// 	}
					// }
					MPI_Irecv(&alpha[activeSize],tranNum,MPI_DOUBLE,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[0]);//MPI_STATUS_IGNORE);
					MPI_Irecv(&G[activeSize],tranNum,MPI_DOUBLE,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[1]);//MPI_STATUS_IGNORE);
					MPI_Irecv(&xSquare[activeSize],tranNum,MPI_DOUBLE,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[2]);//MPI_STATUS_IGNORE);
					MPI_Irecv(&labels[activeSize],tranNum,MPI_INT,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[3]);//MPI_STATUS_IGNORE);
					MPI_Irecv(&globalIndex[activeSize],tranNum,MPI_INT,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[4]);//MPI_STATUS_IGNORE);
					MPI_Irecv(&features[activeSize*(maxId+1)],tranNum*(maxId+1),MPI_DOUBLE_INT,activeNum_[end].rank,0,MPI_COMM_WORLD,&requestRecv[5]);//MPI_STATUS_IGNORE);
					double *tranCache = cache->getTran();
					int size = (int)cache->getSize();
					MPI_Recv(tranCache,size*tranNum,MPI_DOUBLE,activeNum_[end].rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					cache->setTranData(activeSize,tranNum);
					activeSize = activeNum_[begin].active;
					MPI_Waitall(6, &requestRecv[0], &statusRecv[0]);
				}
				if(myRank == activeNum_[end].rank){
					//解决负载平衡后gfmax.rank变化的问题
					// if(myRank == gfmax.rank ){//&& work_i>=activeNum_[end].active){
					// 	//MPI_Send(&work_i,1,MPI_INT,activeNum_[begin].rank,0,MPI_COMM_WORLD);
					// 	//当gfmax的索引i的数据属于要传输的数据时，更改gfmax.rank信息
					// 	if(work_i>=activeNum_[end].active){
					// 		gfmax.rank = activeNum_[begin].rank;
					// 	}
					// }
					MPI_Isend(&alpha[activeNum_[end].active],tranNum,MPI_DOUBLE,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[0]);
					MPI_Isend(&G[activeNum_[end].active],tranNum,MPI_DOUBLE,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[1]);
					MPI_Isend(&xSquare[activeNum_[end].active],tranNum,MPI_DOUBLE,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[2]);
					MPI_Isend(&labels[activeNum_[end].active],tranNum,MPI_INT,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[3]);
					MPI_Isend(&globalIndex[activeNum_[end].active],tranNum,MPI_INT,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[4]);
					MPI_Isend(&features[activeNum_[end].active*(maxId+1)],tranNum*(maxId+1),MPI_DOUBLE_INT,activeNum_[begin].rank,0,MPI_COMM_WORLD,&requestSend[5]);
					activeSize = activeNum_[end].active;
					double *tranCache = cache->getTranData(activeSize,tranNum);
					int size = (int)cache->getSize();
					MPI_Send(tranCache,size*tranNum,MPI_DOUBLE,activeNum_[begin].rank,0,MPI_COMM_WORLD);
					MPI_Waitall(6, &requestSend[0], &statusSend[0]);
				}
				begin++;
				end--;
			}
			else
				break;
		}
		quick_sort(activeNum_,0,commSz-1);
	}

	gfmax.rank = LBrank;
	if(myRank == gfmax.rank)
		work_i = LBi;
	// for(int i=0;i<commSz;i++){
	// 	sumActive+=activeNum_[i].active;
	// }
	// MPI_Allgather(&activeSize,&maxActive,1,MPI_INT,MPI_MAX,3,MPI_COMM_WORLD);
}

void Solver::Solve2_mpi(Model &model, const SvmParameter &param){
    int commSz;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Request RequestA,RequestI,RequestD;
	MPI_Status StatusA,StatusI,StatusD;

    MPI_Datatype type[2] = {MPI_DOUBLE,MPI_INT};
    MPI_Aint disp[2] = {0,sizeof(double)*3};
    int block[2] = {3,2};
    MPI_Datatype MPI_DATA_INFO;
    MPI_Type_create_struct(2,block,disp,type,&MPI_DATA_INFO);
    MPI_Type_commit(&MPI_DATA_INFO);

    alpha = new double[numLocalSamples];
    G = new double[numLocalSamples];
    alpha_status = new char[numLocalSamples];
	shrunk_status = new int[numLocalSamples];
	activeNum = new int[commSz];
	activeNum_ = new activeInfo[commSz];
	activeSize = numLocalSamples;

	shrunk_in_cache = new int[numLocalSamples];
	Allshrunk_in_cache = new int[numAllSamples];

	recSamples = new int[commSz];
	s_displs = new int[commSz];

	// if(myRank == 0){
	// // 	ATLabels = new int[numAllSamples/10];
	// // 	ATSquares = new double[numAllSamples/10];
	// // 	ATAlpha = new double[numAllSamples/10];
	// // 	ATG = new double[numAllSamples/10];
	// // 	ATGlobalIndex = new int[numAllSamples/10];
	// // 	ATFeatures = new SvmFeature[numAllSamples/10*(maxId+1)];
	// 	recValues = new int[commSz];
	// 	v_displs = new int[commSz];
	// 	recSamples = new int[commSz];
	// 	s_displs = new int[commSz];
	// // 	sendSamples = new int[commSz];
	// // 	sendValues = new int[commSz];
	// }

    //cout<<"localstart"<<dataset.localStart<<endl;
    for(int i=0;i<numLocalSamples;i++){
        alpha[i] = 0;
		G[i] = -1;	//*p:minus_ones[i] = -1
        //update_alpha_status(i);
		shrunk_status[i] = 0;
	}

    iter = 0;
	int max_iter = max(10000000, numAllSamples>INT_MAX/100 ? INT_MAX : 100*numAllSamples);
	//int counter = min(numAllSamples,1000)+1;
	int counter = commIter;
	int counter2 = commIter;
	sampleMax = new SvmFeature[maxId+1]();
	sampleMin = new SvmFeature[maxId+1]();

	if(myRank == 0){
		cout<<"mpi size: "<<commSz<<endl;
		cout<<"maxid: "<<maxId<<endl;
    	cout << endl<< "____________________________________" << endl;
		cout<<"Training... "<<endl;
		cout<<"The training process stops when error is less than eps. "<<endl;
	}

	struct timeval start, end; 
	float time_use=0;

	MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::train.Start();
    while(iter < max_iter){
		// if(myRank == 0&&iter>3000){
		// 	cout<<"iter000: "<<iter<<endl;
		// }
		//各节点选择违反KKT条件最严重的一对变量
		//MPI_Barrier(MPI_COMM_WORLD);
		// if(counter == 1000 && myRank==0)
		// 	cout<<"101010 "<<endl;

		TimeProfile::cpSelect.Start();
		int i,j;
		if(select_working_set2(i,j)!=0){
				break;
		}
		if(i==-1)
			fmax.f = -INF;
		else
        	fmax.f = -G[i]*labels[i];
			//fmax.f = -G[i]*labels[i] + cache_status[i]*0.1*C;
        fmax.rank = myRank;
		//fmax.rank = i*commSz+myRank;
		if(j==-1)
			fmin.f = INF;
		else
        	fmin.f = -G[j]*labels[j];
        fmin.rank = myRank;
		TimeProfile::cpSelect.End();
		MPI_Barrier(MPI_COMM_WORLD);
		// if(myRank == 0&&iter>3000){
		// 	cout<<"iter001: "<<iter<<endl;
		// }
		// if(iter>3555){
		// 	cout<<"rank i j"<<myRank<<" "<<i<<" "<<j<<endl;
		// }
		// if(counter == 1000 && myRank==0)
		// 	cout<<"999 "<<endl;
		//得到全局违反KKT条件最严重的一对变量
		TimeProfile::cmSelect.Start();
        MPI_Allreduce(&fmax,&gfmax,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);
        MPI_Allreduce(&fmin,&gfmin,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
		TimeProfile::cmSelect.End();

		// gfmaxI = gfmax.rank/commSz;
		// gfmax.rank = gfmax.rank%commSz;
		// if(myRank == gfmax.rank && i!=gfmaxI)
		// 	cout<<"wrong"<<endl;
		//MPI_Barrier(MPI_COMM_WORLD);
		// if(myRank == 10 && iter>75000 && iter<76000)
		// {
		// 	//cout<<"G: "<<gfmax.f<<" "<<gfmin.f<<endl;
		// 	cout<<"time: "<<(double)(qwe1-asd1)/CLOCKS_PER_SEC<<endl;
		// }
		// if(counter == 1000 && myRank==0)
		// 	cout<<"101010 "<<endl;
		if(iter%1000 == 0 && myRank == 0){
			cout<<"iter: "<<iter<<"	error: "<<gfmax.f-gfmin.f<<endl;
			//cout<<"iter: "<<iter<<"	error: "<<-infoMax.G_*infoMax.y_-gfmin.f<<endl;
		}
		if(gfmax.f-gfmin.f<eps){
			//cout<<gfmax.f<<" "<<gfmin.f<<endl;
			rho = (gfmax.f+gfmin.f)/2;
			if(myRank == 0)
				cout<<"number of iteration: "<<iter<<endl;
			//cout<<rho<<endl;
			break;
		}
		// if(myRank==0){
		// 	if(shrunk_status[0]!=0)
		// 	cout<<"shrink status: "<<shrunk_status[0]<<endl;
		// }
		
		TimeProfile::cpShrinking.Start();
		if(shrinking == 1)
			update_shrunk(gfmax.f, gfmin.f);
		if(--counter == 0){
			//MPI_gather(&activeSize,&maxActive,1,MPI_INT,MPI_MAX,3,MPI_COMM_WORLD);
			//if(myRank == 3)
				//cout<<"load balance: "<<maxActive-minActive<<" "<<activeSize<<" "<<maxActive<<endl;
			//cout<<"rank: "<<myRank<<" "<<activeSize<<endl;
			counter = commIter;//min(activeSize,1000);
			if(shrinking > 0) do_shrinking(i,j);
			//if(myRank == 0) cout<<"."<<endl;
		}
		TimeProfile::cpShrinking.End();

		TimeProfile::cmLoadBalance.Start();
		if(--counter2 == 0){
			counter2 = commIter;
			//if( activeNum_[commSz-1].active-activeNum_[0].active > 20){//numAllSamples/commSz*0.08){
			//if(activeNum_[commSz-1].active - avgActive > 10){
			if(loadbalance == 1)
				load_balance2(i);
			//}
		}
		TimeProfile::cmLoadBalance.End();

		//MPI_Barrier(MPI_COMM_WORLD);
		// if(myRank == 0&&iter==3556){
		// 	cout<<"i j"<<i<<" "<<j<<endl;
		// }
		//std::cout<<"Gmax_idx Gmin_idx: "<<i<<" "<<j<<std::endl;
		//cout<<"Gmax Gmin: "<<gfmax.f<<" "<<gfmin.f<<endl;
		++iter;
		

		//判断是否满足停机条件
        // if(gfmax.f-gfmin.f<eps){
		// 	//cout<<gfmax.f<<" "<<gfmin.f<<endl;
		// 	rho = (gfmax.f+gfmin.f)/2;
		// 	if(myRank == 0)
		// 		cout<<"number of iteration: "<<iter<<endl;
		// 	//cout<<rho<<endl;
		// 	break;
		// }
	
		//获取gfmax数据信息
		//TimeProfile::cpSample.Start();
        if(myRank == gfmax.rank){
            infoMax.alpha_ = alpha[i];
            infoMax.xSquare_ = xSquare[i];
			infoMax.G_ = G[i];
            infoMax.y_ = labels[i];
			infoMax.globalIndex_ = globalIndex[i];
			//memcpy(sampleMax,&features[index[i]],sizeof(SvmFeature)*(index[i+1]-index[i]));
        }
		//TimeProfile::cpSample.End();

		MPI_Barrier(MPI_COMM_WORLD);
		//广播gfmax数据信息
		TimeProfile::cmSample.Start();
		// if(myRank == 0&&iter>3000){
		// 	cout<<"iter111: "<<iter<<endl;
		// }
        MPI_Bcast(&infoMax,1,MPI_DATA_INFO,gfmax.rank,MPI_COMM_WORLD);
		// if(counter == 1000 && myRank==0)
		// 	cout<<"111 "<<endl;
		// if(myRank == 0&&iter>3000){
		// 	cout<<"iter222: "<<iter<<endl;
		// }
		if(cache_status[infoMax.globalIndex_]==0){
			cache_status[infoMax.globalIndex_]=1;
			if(myRank == gfmax.rank)
				//memcpy(sampleMax,&features[index[i]],sizeof(SvmFeature)*(index[i+1]-index[i]));
				memcpy(sampleMax,&features[index[i]],sizeof(SvmFeature)*(maxId+1));
			MPI_Bcast(sampleMax,maxId+1,MPI_DOUBLE_INT,gfmax.rank,MPI_COMM_WORLD);
		}
		TimeProfile::cmSample.End();

		//MPI_Barrier(MPI_COMM_WORLD);
		// if(counter == 1000 && myRank==0)
		// 	cout<<"222 "<<endl;

		TimeProfile::cpKernel.Start();
		gettimeofday(&start,NULL);
		//double *Q_max = get_Qmax(infoMax.globalIndex_,numLocalSamples);
		double *Q_max = get_Qmax(infoMax.globalIndex_,activeSize);
		gettimeofday(&end,NULL);
		TimeProfile::cpKernel.End();
		time_use+=(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec)/1000000;//秒
        //MPI_Bcast(sampleMax,maxId,MPI_DOUBLE_INT,gfmax.rank,MPI_COMM_WORLD);
		// if(counter == 1000 && myRank==0)
		// 	cout<<"333 "<<endl;
		//MPI_Barrier(MPI_COMM_WORLD);

		//根据gfmax以及二阶梯度选择fmin
		TimeProfile::cpSelect1.Start();
		double obj_min;
		selectWorkingSet2(j,obj_min,Q_max);
		if(j==-1)
			fmin.f = INF;
		else
        	fmin.f = obj_min;
        fmin.rank = myRank;
		TimeProfile::cpSelect1.End();
		// if(counter == 1000 && myRank==0)
		// 	cout<<"444 "<<endl;
		MPI_Barrier(MPI_COMM_WORLD);
		//得到全局gfmin
		 
		TimeProfile::cmSelect1.Start();
		MPI_Allreduce(&fmin,&gfmin,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
		TimeProfile::cmSelect1.End();
		// if(counter == 1000 && myRank==0)
		// 	cout<<"555 "<<endl;
		//MPI_Barrier(MPI_COMM_WORLD);

		//获取gfmin数据信息
		//TimeProfile::cpSample.Start();
		if(myRank == gfmin.rank){
            infoMin.alpha_ = alpha[j];
			infoMin.xSquare_ = xSquare[j];
			infoMin.G_ = G[j];
            infoMin.y_ = labels[j];
			infoMin.globalIndex_ = globalIndex[j];
			//memcpy(sampleMin,&features[index[j]],sizeof(SvmFeature)*(index[j+1]-index[j]));
        }
		//TimeProfile::cpSample.End();

		//广播gfmin数据信息
		TimeProfile::cmSample.Start();
        MPI_Bcast(&infoMin,1,MPI_DATA_INFO,gfmin.rank,MPI_COMM_WORLD);
		if(cache_status[infoMin.globalIndex_]==0){
			cache_status[infoMin.globalIndex_]=1;
			if(myRank == gfmin.rank)
				//memcpy(sampleMin,&features[index[j]],sizeof(SvmFeature)*(index[j+1]-index[j]));
				memcpy(sampleMin,&features[index[j]],sizeof(SvmFeature)*(maxId+1));
			MPI_Bcast(sampleMin,maxId+1,MPI_DOUBLE_INT,gfmin.rank,MPI_COMM_WORLD);
		}
		TimeProfile::cmSample.End();

		//MPI_Barrier(MPI_COMM_WORLD);
		// if(counter == 1000 && myRank==0)
		// 	cout<<"666 "<<endl;
		TimeProfile::cpKernel.Start();
		gettimeofday(&start, NULL); 
		//double *Q_min = get_Qmin(infoMin.globalIndex_,numLocalSamples);
		double *Q_min = get_Qmin(infoMin.globalIndex_,activeSize);
		gettimeofday(&end,NULL);
        //MPI_Bcast(sampleMin,maxId,MPI_DOUBLE_INT,gfmin.rank,MPI_COMM_WORLD);
		TimeProfile::cpKernel.End();
		time_use+=(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec)/1000000;//微秒
		// if(counter == 1000 && myRank==0)
		// 	cout<<"777 "<<endl;
		//MPI_Barrier(MPI_COMM_WORLD);

		double delta_alpha_max;
		double delta_alpha_min;
		// 更新 alpha[i] and alpha[j]
		//MPI_Barrier(MPI_COMM_WORLD);
		TimeProfile::cpAlpha.Start();
	if(myRank == gfmax.rank){
		double old_alpha_i = infoMax.alpha_;
		double old_alpha_j = infoMin.alpha_;
		if(infoMax.y_!=infoMin.y_){
			//double quad_coef = 2-2*calKernel(i,j);
			//double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			double quad_coef = 2-2*Q_min[i];
			if (quad_coef <= 0)
				quad_coef = TAU;
			 double delta = (-infoMax.G_-infoMin.G_)/quad_coef;
			//double delta = infoMax.y_*(gfmax.f-gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
            // std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)calKernel(sampleMax,sampleMin)<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double diff = old_alpha_i - old_alpha_j;
			infoMax.alpha_ += delta;
			infoMin.alpha_ += delta;

			if(diff > 0)
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = diff;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = -diff;
				}
			}
			if(diff > 0)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = C - diff;
				}
			}
			else
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = C + diff;
				}
			}
		}
		else
		{
			//double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			double quad_coef = 2-2*Q_min[i];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (infoMax.G_-infoMin.G_)/quad_coef;
			//double delta = infoMax.y_*(-gfmax.f+gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
			// std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)calKernel(sampleMax,sampleMin)<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double sum = old_alpha_i + old_alpha_j;
			infoMax.alpha_ -= delta;
			infoMin.alpha_ += delta;

			if(sum > C)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = sum;
				}
			}
			if(sum > C)
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = sum;
				}
			}
		}

		// 得到alpha增量
		delta_alpha_max = infoMax.alpha_ - old_alpha_i;
		delta_alpha_min = infoMin.alpha_ - old_alpha_j;

        alpha[i] = infoMax.alpha_;
		//update_alpha_status(i);
        
	}
		//MPI_Barrier(MPI_COMM_WORLD);
		TimeProfile::cpAlpha.End();

		MPI_Barrier(MPI_COMM_WORLD);

		TimeProfile::cmDeltaA.Start();
		MPI_Bcast(&delta_alpha_max,1,MPI_DOUBLE,gfmax.rank,MPI_COMM_WORLD);
		MPI_Bcast(&delta_alpha_min,1,MPI_DOUBLE,gfmax.rank,MPI_COMM_WORLD);
		TimeProfile::cmDeltaA.End();
		// if(counter == 1000 && myRank==0)
		// 	cout<<"888 "<<endl;
        if(myRank == gfmin.rank){
            alpha[j] = alpha[j] + delta_alpha_min;
			//update_alpha_status(j);
        }

		//MPI_Barrier(MPI_COMM_WORLD);

		//更新梯度
		TimeProfile::cpGradient.Start();
		//for(int k=0;k<numLocalSamples;k++){
		int k;
//#pragma omp parallel for //private(k) schedule(guided)
		for(k=0;k<activeSize;k++){
			G[k] += Q_max[k]*labels[k]*infoMax.y_*delta_alpha_max + Q_min[k]*labels[k]*infoMin.y_*delta_alpha_min;
        }
		TimeProfile::cpGradient.End();

		//MPI_Barrier(MPI_COMM_WORLD);

		
	}
	MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::train.End();

	if(myRank == 0){
		cout<<"kernel time: "<<time_use<<endl;
		std::cout<<"access hitLen hitPar: "<<access<<" "<<hitLen<<" "<<hitPar<<" "<<std::endl;
		std::cout<<"hit rate: "<<(float)hitLen/access*100<<"% "<<(float)hitPar/access*100<<"%"<<std::endl;
	}

	//判断支持向量
    //model.CheckSupportVector_mpi(maxId, alpha, features, labels, index, numLocalSamples, param, rho);
	model.CheckSupportVector_mpi(maxId, alpha, features, labels, index, activeSize, param, rho);

    delete[] G;
    delete[] alpha_status;
	delete[] sampleMax;
	delete[] sampleMin;
	delete[] xSquare;
	delete[] QD;
	delete cache;
	delete[] alpha;
    delete[] features;
    delete[] labels;
    delete[] index;

	delete[] recSamples;
	delete[] s_displs;

    if(myRank==0){
        delete[] allFeatures;
        delete[] allLabels;
        delete[] allIndex;
    }
}

void Solver::Solve0_mpi(Model &model, const SvmParameter &param){

    int commSz;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Request RequestA,RequestI,RequestD;
	MPI_Status StatusA,StatusI,StatusD;

    MPI_Datatype type[2] = {MPI_DOUBLE,MPI_INT};
    MPI_Aint disp[2] = {0,sizeof(double)*3};
    int block[2] = {3,2};
    MPI_Datatype MPI_DATA_INFO;
    MPI_Type_create_struct(2,block,disp,type,&MPI_DATA_INFO);
    MPI_Type_commit(&MPI_DATA_INFO);

    alpha = new double[numLocalSamples];
    G = new double[numLocalSamples];
    alpha_status = new char[numLocalSamples];

    //cout<<"localstart"<<dataset.localStart<<endl;
    for(int i=0;i<numLocalSamples;i++){
        alpha[i] = 0;
		G[i] = -1;	//*p:minus_ones[i] = -1
        update_alpha_status(i);
		//cache_status[i] = 0;
	}

    iter = 0;
	int max_iter = max(10000000, numAllSamples>INT_MAX/100 ? INT_MAX : 100*numAllSamples);
	//int counter = min(numAllSamples,1000)+1;
	int counter = 1000;
	sampleMax = new SvmFeature[maxId+1]();
	sampleMin = new SvmFeature[maxId+1]();
	//SvmFeature *selectedSample = new SvmFeature[2*maxId+2]();

	if(myRank == 0){
    	cout << endl<< "____________________________________" << endl;
		cout<<"Training... "<<endl;
		cout<<"The training process stops when error is less than eps. "<<endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	TimeProfile::train.Start();
    while(iter < max_iter)
	{
		// show progress and do shrinking
		// counter--;
        // if(counter == 0 && myRank == 0)
		// {
		// 	//counter = min(numAllSamples,1000);
		// 	counter = 1000;
		// 	//if(shrinking) do_shrinking();
		// 	printf(".");
		// 	fflush(stdout);
		// }
		
		//cpSelect.start = MPI_Wtime();
		TimeProfile::cpSelect.Start();
		int i,j;
		if(select_working_set2(i,j)!=0)
		{
				break;
		}
		if(i==-1)
			fmax.f = -INF;
		else
        	fmax.f = -G[i]*labels[i];
        fmax.rank = myRank;
		if(j==-1)
			fmin.f = INF;
		else
        	fmin.f = -G[j]*labels[j];
        fmin.rank = myRank;
		TimeProfile::cpSelect.End();
		
		//gfmax=fmax;
		//gfmin=fmin;

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cmSelect.Start();
        MPI_Allreduce(&fmax,&gfmax,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);
        MPI_Allreduce(&fmin,&gfmin,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
		TimeProfile::cmSelect.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		// if(iter>11975&&iter<11985&&myRank==gfmax.rank){
		// //if((iter == 11981||iter == 11980)&&myRank==gfmax.rank){
		// 	cout<<"notion: "<<iter<<" "<<myRank<< endl;
		// 	std::cout<<"Gmax_idx Gmin_idx: "<<i<<" "<<j<<std::endl;
		//  	cout<<"Gmax Gmin: "<<gfmax.f<<" "<<gfmin.f<<endl;
		// 	 cout<<gfmax.rank<<" "<<G[i]<<endl;
			
		// 	//exit(1);
		// }
		//std::cout<<"Gmax_idx Gmin_idx: "<<i<<" "<<j<<std::endl;
		//cout<<"Gmax Gmin: "<<gfmax.f<<" "<<gfmin.f<<endl;
		++iter;
		if(iter%1000 == 0&& myRank == 0){
			cout<<"iter: "<<iter<<"	error: "<<gfmax.f-gfmin.f<<endl;
			//cout<<gfmax.f<<" "<<gfmin.f<<" "<<gfmax.f-gfmin.f<<endl;
		}
        if(gfmax.f-gfmin.f<eps){
			//cout<<gfmax.f<<" "<<gfmin.f<<endl;
			rho = (gfmax.f+gfmin.f)/2;
			if(myRank == 0)
				cout<<"number of iteration: "<<iter<<endl;
			//cout<<rho<<endl;
			break;
		}

		TimeProfile::cpSample.Start();
        if(myRank == gfmax.rank){
            infoMax.alpha_ = alpha[i];
            infoMax.xSquare_ = xSquare[i];
            infoMax.y_ = labels[i];
			infoMax.G_ = G[i];
			infoMax.globalIndex_ = globalIndex[i];
            //infoMax.length_ = index[i+1]-index[i];
            //sampleMax = features + index[i];
			// for(int q=index[i];q<index[i+1];q++){
			// 	sampleMax[q-index[i]] = features[q];
			// }
			memcpy(sampleMax,&features[index[i]],sizeof(SvmFeature)*(index[i+1]-index[i]));
        }
        if(myRank == gfmin.rank){
            infoMin.alpha_ = alpha[j];
			infoMin.xSquare_ = xSquare[j];
            infoMin.y_ = labels[j];
			infoMin.G_ = G[j];
			infoMin.globalIndex_ = globalIndex[j];
            //infoMin.length_ = index[j+1]-index[j];
            //sampleMin = features + index[j];
			// for(int q=index[j];q<index[j+1];q++){
			// 	sampleMin[q-index[j]] = features[q];
			// }
			memcpy(sampleMin,&features[index[j]],sizeof(SvmFeature)*(index[j+1]-index[j]));
        }
		TimeProfile::cpSample.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cmSample.Start();
        MPI_Bcast(&infoMax,1,MPI_DATA_INFO,gfmax.rank,MPI_COMM_WORLD);
		MPI_Bcast(&infoMin,1,MPI_DATA_INFO,gfmin.rank,MPI_COMM_WORLD);
		MPI_Bcast(sampleMax,maxId,MPI_DOUBLE_INT,gfmax.rank,MPI_COMM_WORLD);
		MPI_Bcast(sampleMin,maxId,MPI_DOUBLE_INT,gfmin.rank,MPI_COMM_WORLD);
		TimeProfile::cmSample.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();
		
		// update alpha[i] and alpha[j], handle bounds carefully
    	//if(myRank == 0){
		TimeProfile::cpAlpha.Start();
		double old_alpha_i = infoMax.alpha_;
		double old_alpha_j = infoMin.alpha_;
        //double old_alpha_i = alpha[i];
		//double old_alpha_j = alpha[j];

		if(infoMax.y_!=infoMin.y_)
        //if(labels[i]!=labels[j])
		{
			//double quad_coef = 2-2*calKernel(i,j);
			double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			if (quad_coef <= 0)
				quad_coef = TAU;
			// double delta = (-G[i]-G[j])/quad_coef;
			double delta = infoMax.y_*(gfmax.f-gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
            // std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)calKernel(sampleMax,sampleMin)<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double diff = old_alpha_i - old_alpha_j;
			infoMax.alpha_ += delta;
			infoMin.alpha_ += delta;

			if(diff > 0)
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = diff;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = -diff;
				}
			}
			if(diff > 0)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = C - diff;
				}
			}
			else
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = C + diff;
				}
			}
		}
		else
		{
			double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = infoMax.y_*(-gfmax.f+gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
			// std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)calKernel(sampleMax,sampleMin)<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double sum = old_alpha_i + old_alpha_j;
			infoMax.alpha_ -= delta;
			infoMin.alpha_ += delta;

			if(sum > C)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = sum;
				}
			}
			if(sum > C)
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = sum;
				}
			}
		}
		if(myRank == gfmax.rank){
            alpha[i] = infoMax.alpha_;
			update_alpha_status(i);
        }
        if(myRank == gfmin.rank){
            alpha[j] = infoMin.alpha_;
			update_alpha_status(j);
        }
		
		// 得到alpha增量
		double delta_alpha_i = infoMax.alpha_ - old_alpha_i;
		double delta_alpha_j = infoMin.alpha_ - old_alpha_j;
		TimeProfile::cpAlpha.End();
    //}
		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cpGradient.Start();
		//更新梯度
		for(int k=0;k<numLocalSamples;k++){
			G[k] += calMaxKernel(k,sampleMax)*labels[k]*infoMax.y_*delta_alpha_i + calMinKernel(k,sampleMin)*labels[k]*infoMin.y_*delta_alpha_j;
			//G[k] += Q_max[k]*labels[k]*infoMax.y_*delta_alpha_i + Q_min[k]*labels[k]*infoMin.y_*delta_alpha_j;
			
			// if(myRank==1&& iter == 11982&&k<210&&k>190){
			// 	//cout<<iter<<" "<<k<<" "<<G[k]<<endl;
			// 	//if(G[k]>2||G[k]<-2){
			// 	//cout<<gamma * (xSquare[k] - 2 * innerProduct(k, sampleMax) + infoMax.xSquare_)<<endl;
			// 	cout<<iter<<" "<<k<<" "<<G[k]<<":"<<calMaxKernel(k,sampleMax)<<" "<<delta_alpha_i<<" "<<calMinKernel(k,sampleMin)<<" "<<delta_alpha_j<<endl;
			// 	//}
			// }
		}
		TimeProfile::cpGradient.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();
		
		// update alpha_status and G_bar

		
			//  bool ui = is_upper_bound(i);
			//  bool uj = is_upper_bound(j);
			 //update_alpha_status(i);
			 //update_alpha_status(j);
		
	}
	TimeProfile::train.End();
	// MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */


    // localAlpha = new double[commIter];
    // globalAlpha = new double[commIter*commSz];
    // localIndex = new int[commIter];
    // globalIndex = new int[commIter*commSz];
    // localData = new SvmSample[commIter];
    // globalData = new SvmSample[commIter*commSz];

        // if(iter % commIter == 0){
        //     if(iter != commIter){
        //         MPI_Wait(&RequestA, &StatusA);
        //         MPI_Wait(&RequestD, &StatusD);
        //         MPI_Wait(&RequestI, &StatusI);
        //     }
        //     MPI_Iallgather(localAlpha,commIter,MPI_DOUBLE,globalAlpha,commIter,MPI_DOUBLE,MPI_COMM_WORLD,RequestA);
        //     MPI_Iallgather(localIndex,commIter,MPI_INT,globalIndex,commIter,MPI_INT,MPI_COMM_WORLD,RequestI);
        //     MPI_Iallgather(localData,commIter,MPI_DOUBLE,globalData,commIter,MPI_DOUBLE,MPI_COMM_WORLD,RequestD);
        // }
    
	//int df;
	//MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allreduce(&myRank,&df,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    //cout<<df;
    model.CheckSupportVector_mpi(maxId, alpha, features, labels, index, numLocalSamples, param, rho);
	// for(int k=0;k<1;k++){
    //     if(myRank == 0)
    //         cout << "rank"<<myRank<< "alpha "<<k<<":"<<labels[k]*alpha[k] <<" "<<features[index[k]].value<<endl;
	// 	//if(myRank == 1)
    //         //cout << "rank"<<myRank<< "alpha "<<k<<":"<<labels[k]*alpha[k] <<" "<<features[index[k]].value<<endl;
	// }
	//if(myRank==0)
	//cout<<"@"<<endl;
    //delete[] alpha;
    delete[] G;
    //delete[] y;
    delete[] alpha_status;
	delete[] sampleMax;
	delete[] sampleMin;
	delete[] xSquare;
	delete[] QD;
	delete cache;
//if(myRank==0)
	//cout<<"@"<<endl;
	delete[] alpha;
    delete[] features;
    delete[] labels;
    delete[] index;
    if(myRank==0){
        delete[] allFeatures;
        delete[] allLabels;
        delete[] allIndex;
    }
	//if(myRank==0)
	//cout<<"@"<<endl;
}

void Solver::Solve_mpi(Model &model, const SvmParameter &param){

    int commSz;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Request RequestA,RequestI,RequestD;
	MPI_Status StatusA,StatusI,StatusD;

    MPI_Datatype type[2] = {MPI_DOUBLE,MPI_INT};
    MPI_Aint disp[2] = {0,sizeof(double)*3};
    int block[2] = {3,2};
    MPI_Datatype MPI_DATA_INFO;
    MPI_Type_create_struct(2,block,disp,type,&MPI_DATA_INFO);
    MPI_Type_commit(&MPI_DATA_INFO);

    alpha = new double[numLocalSamples];
    G = new double[numLocalSamples];
    alpha_status = new char[numLocalSamples];

    //cout<<"localstart"<<dataset.localStart<<endl;
    for(int i=0;i<numLocalSamples;i++){
        alpha[i] = 0;
		G[i] = -1;	//*p:minus_ones[i] = -1
        update_alpha_status(i);
		//cache_status[i] = 0;
	}

    iter = 0;
	int max_iter = max(10000000, numAllSamples>INT_MAX/100 ? INT_MAX : 100*numAllSamples);
	//int counter = min(numAllSamples,1000)+1;
	int counter = 1000;
	sampleMax = new SvmFeature[maxId+1]();
	sampleMin = new SvmFeature[maxId+1]();
	//SvmFeature *selectedSample = new SvmFeature[2*maxId+2]();

	// if(myRank == 1){
    // 	for(int i=0;i<numLocalSamples;i++){
	// 		cout<<globalIndex[i]<<" ";
	// 	}
	// }

	if(myRank == 0){
    	cout << endl<< "____________________________________" << endl;
		cout<<"Training... "<<endl;
		cout<<"The training process stops when error is less than eps. "<<endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	TimeProfile::train.Start();
    while(iter < max_iter)
	{
		// show progress and do shrinking
		// counter--;
        // if(counter == 0 && myRank == 0)
		// {
		// 	//counter = min(numAllSamples,1000);
		// 	counter = 1000;
		// 	//if(shrinking) do_shrinking();
		// 	printf(".");
		// 	fflush(stdout);
		// }
		
		//cpSelect.start = MPI_Wtime();
		TimeProfile::cpSelect.Start();
		int i,j;
		if(select_working_set2(i,j)!=0)
		{
				break;
		}
		if(i==-1)
			fmax.f = -INF;
		else
        	fmax.f = -G[i]*labels[i];
        fmax.rank = myRank;
		if(j==-1)
			fmin.f = INF;
		else
        	fmin.f = -G[j]*labels[j];
        fmin.rank = myRank;
		TimeProfile::cpSelect.End();
		
		//gfmax=fmax;
		//gfmin=fmin;

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cmSelect.Start();
        MPI_Allreduce(&fmax,&gfmax,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);
        MPI_Allreduce(&fmin,&gfmin,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
		TimeProfile::cmSelect.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		 //std::cout<<"Gmax_idx Gmin_idx: "<<i<<" "<<j<<std::endl;
		 //cout<<"Gmax Gmin: "<<gfmax.f<<" "<<gfmin.f<<endl;
		++iter;
		if(iter%1000 == 0&& myRank == 0){
			cout<<"iter: "<<iter<<"	error: "<<gfmax.f-gfmin.f<<endl;
			//cout<<gfmax.f<<" "<<gfmin.f<<" "<<gfmax.f-gfmin.f<<endl;
		}
        if(gfmax.f-gfmin.f<eps){
			//cout<<gfmax.f<<" "<<gfmin.f<<endl;
			rho = (gfmax.f+gfmin.f)/2;
			if(myRank == 0)
				cout<<"number of iteration: "<<iter<<endl;
			//cout<<rho<<endl;
			break;
		}

		//TimeProfile::cpSample.Start();
        if(myRank == gfmax.rank){
            infoMax.alpha_ = alpha[i];
            infoMax.xSquare_ = xSquare[i];
			infoMax.G_ = G[i];
            infoMax.y_ = labels[i];
			infoMax.globalIndex_ = globalIndex[i];
            //infoMax.length_ = index[i+1]-index[i];
            //sampleMax = features + index[i];
			// for(int q=index[i];q<index[i+1];q++){
			// 	sampleMax[q-index[i]] = features[q];
			// }
        }
        if(myRank == gfmin.rank){
            infoMin.alpha_ = alpha[j];
			infoMin.xSquare_ = xSquare[j];
			infoMin.G_ = G[j];
            infoMin.y_ = labels[j];
			infoMin.globalIndex_ = globalIndex[j];
            //infoMin.length_ = index[j+1]-index[j];
            //sampleMin = features + index[j];
			// for(int q=index[j];q<index[j+1];q++){
			// 	sampleMin[q-index[j]] = features[q];
			// }
        }
		//TimeProfile::cpSample.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cmSample.Start();
        MPI_Bcast(&infoMax,1,MPI_DATA_INFO,gfmax.rank,MPI_COMM_WORLD);
		MPI_Bcast(&infoMin,1,MPI_DATA_INFO,gfmin.rank,MPI_COMM_WORLD);

		if(cache_status[infoMax.globalIndex_]==0){
			cache_status[infoMax.globalIndex_]=1;
			if(myRank == gfmax.rank)
				memcpy(sampleMax,&features[index[i]],sizeof(SvmFeature)*(index[i+1]-index[i]));
			MPI_Bcast(sampleMax,maxId+1,MPI_DOUBLE_INT,gfmax.rank,MPI_COMM_WORLD);
		}
		TimeProfile::cmSample.End();
		
		TimeProfile::cpKernel.Start();
		double *Q_max = get_Qmax(infoMax.globalIndex_,numLocalSamples);
		TimeProfile::cpKernel.End();

		TimeProfile::cmSample.Start();
		if(cache_status[infoMin.globalIndex_]==0){
			cache_status[infoMin.globalIndex_]=1;
			if(myRank == gfmin.rank)
				memcpy(sampleMin,&features[index[j]],sizeof(SvmFeature)*(index[j+1]-index[j]));
			MPI_Bcast(sampleMin,maxId+1,MPI_DOUBLE_INT,gfmin.rank,MPI_COMM_WORLD);
		}
		TimeProfile::cmSample.End();

		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();
		
		// update alpha[i] and alpha[j], handle bounds carefully
		
		TimeProfile::cpKernel.Start();
		double *Q_min = get_Qmin(infoMin.globalIndex_,numLocalSamples);
		TimeProfile::cpKernel.End();

        //double old_alpha_i = alpha[i];
		//double old_alpha_j = alpha[j];
		double delta_alpha_max;
		double delta_alpha_min;

		MPI_Barrier(MPI_COMM_WORLD);
		TimeProfile::cpAlpha.Start();
	if(myRank == gfmax.rank){
		double old_alpha_i = infoMax.alpha_;
		double old_alpha_j = infoMin.alpha_;
		if(infoMax.y_!=infoMin.y_)
        //if(labels[i]!=labels[j])
		{
			//double quad_coef = 2-2*calKernel(i,j);
			//double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			double quad_coef = 2-2*Q_min[i];
			if (quad_coef <= 0)
				quad_coef = TAU;
			// double delta = (-G[i]-G[j])/quad_coef;
			double delta = infoMax.y_*(gfmax.f-gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
            // std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)Q_min[i]<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double diff = old_alpha_i - old_alpha_j;
			infoMax.alpha_ += delta;
			infoMin.alpha_ += delta;

			if(diff > 0)
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = diff;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = -diff;
				}
			}
			if(diff > 0)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = C - diff;
				}
			}
			else
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = C + diff;
				}
			}
		}
		else
		{
			//double quad_coef = 2-2*calKernel(sampleMax,sampleMin);
			double quad_coef = 2-2*Q_min[i];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = infoMax.y_*(-gfmax.f+gfmin.f)/quad_coef;
            // std::cout<<quad_coef<<std::endl;
			// std::cout<<-infoMax.y_*gfmax.f<<" "<<-infoMin.y_*gfmin.f<<std::endl;
			// std::cout<<-1*(float)Q_min[i]<<" "<<std::endl;
            // std::cout<<delta<<std::endl;
			double sum = old_alpha_i + old_alpha_j;
			infoMax.alpha_ -= delta;
			infoMin.alpha_ += delta;

			if(sum > C)
			{
				if(infoMax.alpha_ > C)
				{
					infoMax.alpha_ = C;
					infoMin.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMin.alpha_ < 0)
				{
					infoMin.alpha_ = 0;
					infoMax.alpha_ = sum;
				}
			}
			if(sum > C)
			{
				if(infoMin.alpha_ > C)
				{
					infoMin.alpha_ = C;
					infoMax.alpha_ = sum - C;
				}
			}
			else
			{
				if(infoMax.alpha_ < 0)
				{
					infoMax.alpha_ = 0;
					infoMin.alpha_ = sum;
				}
			}
		}
		delta_alpha_max = infoMax.alpha_ - old_alpha_i;
		delta_alpha_min = infoMin.alpha_ - old_alpha_j;

		alpha[i] = infoMax.alpha_;
		update_alpha_status(i);

	}
		MPI_Barrier(MPI_COMM_WORLD);
		TimeProfile::cpAlpha.End();

		TimeProfile::cmDeltaA.Start();
		MPI_Bcast(&delta_alpha_max,1,MPI_DOUBLE,gfmax.rank,MPI_COMM_WORLD);
		MPI_Bcast(&delta_alpha_min,1,MPI_DOUBLE,gfmax.rank,MPI_COMM_WORLD);
		TimeProfile::cmDeltaA.End();

        if(myRank == gfmin.rank){
            alpha[j] = alpha[j] + delta_alpha_min;
			update_alpha_status(j);
        }
		
		// syn.Start();
		// MPI_Barrier(MPI_COMM_WORLD);
		// syn.End();

		TimeProfile::cpGradient.Start();
		//更新梯度
		for(int k=0;k<numLocalSamples;k++){
			G[k] += Q_max[k]*labels[k]*infoMax.y_*delta_alpha_max + Q_min[k]*labels[k]*infoMin.y_*delta_alpha_min;
		}
		TimeProfile::cpGradient.End();
		
	}
	TimeProfile::train.End();

    // localAlpha = new double[commIter];
    // globalAlpha = new double[commIter*commSz];
    // localIndex = new int[commIter];
    // globalIndex = new int[commIter*commSz];
    // localData = new SvmSample[commIter];
    // globalData = new SvmSample[commIter*commSz];

        // if(iter % commIter == 0){
        //     if(iter != commIter){
        //         MPI_Wait(&RequestA, &StatusA);
        //         MPI_Wait(&RequestD, &StatusD);
        //         MPI_Wait(&RequestI, &StatusI);
        //     }
        //     MPI_Iallgather(localAlpha,commIter,MPI_DOUBLE,globalAlpha,commIter,MPI_DOUBLE,MPI_COMM_WORLD,RequestA);
        //     MPI_Iallgather(localIndex,commIter,MPI_INT,globalIndex,commIter,MPI_INT,MPI_COMM_WORLD,RequestI);
        //     MPI_Iallgather(localData,commIter,MPI_DOUBLE,globalData,commIter,MPI_DOUBLE,MPI_COMM_WORLD,RequestD);
        // }
    
    model.CheckSupportVector_mpi(maxId, alpha, features, labels, index, numLocalSamples, param, rho);

    delete[] G;
    delete[] alpha_status;
	delete[] sampleMax;
	delete[] sampleMin;
	delete[] xSquare;
	delete[] QD;
	delete cache;
	delete[] alpha;
    delete[] features;
    delete[] labels;
    delete[] index;
    if(myRank==0){
        delete[] allFeatures;
        delete[] allLabels;
        delete[] allIndex;
    }
}

void saveTimeInfo(char *timefilename, SvmParameter param, Solver &solve){
	int myrank;
    int commSz;
    //double *alpha = solver.getAlpha();
    //double *alpha = alpha;
	//cout<<"d"<<endl;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	MPI_Reduce(&(TimeProfile::train.sum),&(TimeProfile::train.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpAlpha.sum),&(TimeProfile::cpAlpha.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpGradient.sum),&(TimeProfile::cpGradient.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpKernel.sum),&(TimeProfile::cpKernel.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	//MPI_Reduce(&(TimeProfile::cpSample.sum),&(TimeProfile::cpSample.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpSelect1.sum),&(TimeProfile::cpSelect1.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpSelect.sum),&(TimeProfile::cpSelect.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cpShrinking.sum),&(TimeProfile::cpShrinking.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	//MPI_Reduce(&(TimeProfile::cpGradient.sum),&(TimeProfile::cpGradient.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	MPI_Reduce(&(TimeProfile::cmSelect1.sum),&(TimeProfile::cmSelect1.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cmSelect.sum),&(TimeProfile::cmSelect.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cmDeltaA.sum),&(TimeProfile::cmDeltaA.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cmSample.sum),&(TimeProfile::cmSample.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&(TimeProfile::cmLoadBalance.sum),&(TimeProfile::cmLoadBalance.all),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	if (myrank == 0) { /* use time on master node */

		TimeProfile::train.sum=TimeProfile::train.all/commSz;
		TimeProfile::cpAlpha.sum=TimeProfile::cpAlpha.all/commSz;
		TimeProfile::cpGradient.sum=TimeProfile::cpGradient.all/commSz;
		TimeProfile::cpKernel.sum=TimeProfile::cpKernel.all/commSz;
		//TimeProfile::cpSample.sum=TimeProfile::cpSample.all/commSz;
		TimeProfile::cpSelect1.sum=TimeProfile::cpSelect1.all/commSz;
		TimeProfile::cpSelect.sum=TimeProfile::cpSelect.all/commSz;
		TimeProfile::cpShrinking.sum=TimeProfile::cpShrinking.all/commSz;
		//TimeProfile::cpGradient.sum=TimeProfile::cpGradient.all/commSz;
		TimeProfile::cmSelect1.sum=TimeProfile::cmSelect1.all/commSz;
		TimeProfile::cmSelect.sum=TimeProfile::cmSelect.all/commSz;
		TimeProfile::cmDeltaA.sum=TimeProfile::cmDeltaA.all/commSz;
		TimeProfile::cmSample.sum=TimeProfile::cmSample.all/commSz;
		TimeProfile::cmLoadBalance.sum=TimeProfile::cmLoadBalance.all/commSz;

		ofstream outfile(timefilename);
		outfile << "Num of cores: " << commSz<<endl;
		outfile<<"The parameters of svm:	C:"<<param.hyperParmC<<" eps:"<<param.epsilon<<" gamma:"<<param.gamma<<endl;
		outfile << "Train data number: " << solve.getAllSampleNum()<<endl;
		//outfile.close();

		cout << "------------------------------------" << endl;
        cout<<"Saving timeout file..."<<endl;
        // ofstream outfile;
		// outfile.open("time.out");
		//ofstream outfile(filename, ios::out | ios::app);

		//cout<<"done"<<endl;
		cout << endl<< "____________________________________" << endl;
		outfile << endl<< "____________________________________" << endl;
		// 14 9 13
		cout << setiosflags(ios::left) << setw(20) << "Train time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << TimeProfile::train.sum// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		outfile << setiosflags(ios::left) << setw(20) << "Train time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << TimeProfile::train.sum// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		//outfile <<"Train time(s): " << train.sum<< endl;
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		// 14 9 13
		double computationTime = TimeProfile::cpAlpha.sum+TimeProfile::cpGradient.sum+TimeProfile::cpKernel.sum+TimeProfile::cpSelect.sum+TimeProfile::cpSelect1.sum+TimeProfile::cpShrinking.sum;
		cout << setiosflags(ios::left) << setw(20) << "   Computation time" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << computationTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		outfile << setiosflags(ios::left) << setw(20) << "   Computation time" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << computationTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		//outfile <<"Computation time: " << cpAlpha.sum+cpGradient.sum+cpSample.sum+cpSelect.sum<< endl;
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		string ComputationTime[] = {"      select alpha","      select alpha1", "      compute kernel", "      compute alpha", "      compute gradient", "      shrinking"};
		double ComputationTime1[] = {TimeProfile::cpSelect.sum,TimeProfile::cpSelect1.sum, TimeProfile::cpKernel.sum, TimeProfile::cpAlpha.sum, TimeProfile::cpGradient.sum, TimeProfile::cpShrinking.sum};
		for (int i = 0; i < 6; ++i) {
			cout << setiosflags(ios::left) << setw(20) << ComputationTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << ComputationTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			outfile << setiosflags(ios::left) << setw(20) << ComputationTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << ComputationTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			//outfile <<computationTime[i] <<": "<<computationTime1[i]<<endl;
		}
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		double communicationTime = TimeProfile::cmSelect.sum+TimeProfile::cmSelect1.sum+TimeProfile::cmSample.sum+TimeProfile::cmDeltaA.sum+TimeProfile::cmLoadBalance.sum;
		cout << setiosflags(ios::left) << setw(20) << "   Communication time" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << communicationTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		outfile << setiosflags(ios::left) << setw(20) << "   Communication time" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << communicationTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		//outfile <<"Communication time: " <<cmSelect.sum+cmSample.sum<<endl;
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		string CommunicationTime[] = {"      reduce f","      reduce f1", "      bcast sample","      bcast deltarA","      load balance"};
		double CommunicationTime1[] = {TimeProfile::cmSelect.sum,TimeProfile::cmSelect1.sum, TimeProfile::cmSample.sum, TimeProfile::cmDeltaA.sum, TimeProfile::cmLoadBalance.sum};
		for (int i = 0; i < 5; ++i) {
			cout << setiosflags(ios::left) << setw(20) << CommunicationTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << CommunicationTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			outfile << setiosflags(ios::left) << setw(20) << CommunicationTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << CommunicationTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			//outfile <<CommunicationTime[i] <<": "<<CommunicationTime1[i]<<endl;
		}
		// cout << "------------------------------------" << endl;
		// outfile << "------------------------------------" << endl;
		// cout << setiosflags(ios::left) << setw(20) << "   Synchronization time" << resetiosflags(ios::left) // 用完之后清除
		// 	<< setiosflags(ios::right) << setw(14) << syn.sum// << setw(12) << "Pop.(10K)"
		// 	<< resetiosflags(ios::right) << endl;
		// outfile << setiosflags(ios::left) << setw(20) << "   Synchronization time" << resetiosflags(ios::left) // 用完之后清除
		// 	<< setiosflags(ios::right) << setw(14) << syn.sum// << setw(12) << "Pop.(10K)"
		// 	<< resetiosflags(ios::right) << endl;
		//outfile <<"Communication time: " <<cmSelect.sum+cmSample.sum<<endl;
		//outfile.close();

		//cout<<"done"<<endl;
		cout << endl<< "____________________________________" << endl;
		outfile << endl<< "____________________________________" << endl;
		// 14 9 13
		double readTime = TimeProfile::readAll.sum+TimeProfile::cmScatter.sum;
		cout << setiosflags(ios::left) << setw(20) << "Read time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << readTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		outfile << setiosflags(ios::left) << setw(20) << "Read time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << readTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		//outfile <<"Read time(s): " << TimeProfile::TimeProfile::readAll.sum+cmScatter.sum<< endl;
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		string ReadTime[] = {"      read all data", "      scatter data"};
		double ReadTime1[] = {TimeProfile::readAll.sum, TimeProfile::cmScatter.sum};
		for (int i = 0; i < 2; ++i) {
			cout << setiosflags(ios::left) << setw(20) << ReadTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << ReadTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			outfile << setiosflags(ios::left) << setw(20) << ReadTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << ReadTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			//outfile <<readTime[i] <<": "<<readTime1[i]<<endl;
		}

		cout << endl<< "____________________________________" << endl;
		outfile << endl<< "____________________________________" << endl;
		// 14 9 13
		double saveModelTime = TimeProfile::saveModel.sum+TimeProfile::checkSV.sum+TimeProfile::cmGather.sum;
		cout << setiosflags(ios::left) << setw(20) << "Save model time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << saveModelTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		outfile << setiosflags(ios::left) << setw(20) << "Save model time(s)" << resetiosflags(ios::left) // 用完之后清除
			<< setiosflags(ios::right) << setw(14) << saveModelTime// << setw(12) << "Pop.(10K)"
			<< resetiosflags(ios::right) << endl;
		//outfile <<"Read time(s): " << readAll.sum+cmScatter.sum<< endl;
		cout << "------------------------------------" << endl;
		outfile << "------------------------------------" << endl;
		string SaveModelTime[] = {"      check SV", "      gather SV", "      save model"};
		double SaveModelTime1[] = {TimeProfile::checkSV.sum, TimeProfile::cmGather.sum, TimeProfile::saveModel.sum};
		for (int i = 0; i < 3; ++i) {
			cout << setiosflags(ios::left) << setw(20) << SaveModelTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << SaveModelTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			outfile << setiosflags(ios::left) << setw(20) << SaveModelTime[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(14) << SaveModelTime1[i] //<< setw(10) << pops[i]
				<< resetiosflags(ios::right) << endl;
			//outfile <<readTime[i] <<": "<<readTime1[i]<<endl;
		}

		cout << endl<< "____________________________________" << endl;
		outfile << endl<< "____________________________________" << endl;
        cout<<"all time: "<<TimeProfile::train.sum+readTime+saveModelTime<<endl;
        outfile<<"all time: "<<TimeProfile::train.sum+readTime+saveModelTime<<endl;
		outfile << "iterations: " <<solve.getIterNum()<< endl;
		
		//cout<<"done"<<endl;
		
		outfile.close();
	}
}

void Solver::predict_mpi(Model &model){
	int myrank;
    int commSz;
    //double *alpha = solver.getAlpha();
    //double *alpha = alpha;
	//cout<<"d"<<endl;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	predictLabels = new int[numLocalSamples];
	SVyAlpha = model.getSVyAlpha();
	index2 = model.getSVindex();
	features2 = model.getSVfeatures();
	xSquare2 = model.getSVnorm();
	rho = model.getRho();
	//cout<<"rho"<<rho<<endl;
	int correct=0;
	allCorrect = 0;
	if(myrank == 0){
		cout << "------------------------------------" << endl;
		cout<<"Predicting..."<<endl;
	}
	for(int i=0;i<numLocalSamples;i++){
		if(myrank==0&&i%100==0){
			cout<<i*numProc<<endl;
		}
		//cout<<i<<endl;
		double sum=0;
		for(int j=0;j<model.getSVnum();j++){
			sum+=calKernel(i,j)*SVyAlpha[j];
		}
		sum+=rho;
		if(sum>0)
			predictLabels[i]=1;
		else
			predictLabels[i]=-1;
		if(predictLabels[i]==labels[i])
			correct++;
	}
	MPI_Reduce(&correct,&allCorrect,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(myrank==0){
		cout<<"done"<<endl;
		cout << "------------------------------------" << endl;
		cout<<"Accuracy: "<<(double)allCorrect/numAllSamples<<" ("<<allCorrect<<"/"<<numAllSamples<<")"<<endl;
		cout << "------------------------------------" << endl;
	}

	delete[] SVyAlpha;
	delete[] index2;
    delete[] features2;
	delete[] xSquare2;

	delete[] xSquare;
	delete[] QD;
	delete cache;
	//if(myRank==0)
	//cout<<"@"<<endl;
    delete[] features;
    delete[] labels;
    delete[] index;
    if(myrank==0){
        delete[] allFeatures;
        delete[] allLabels;
        delete[] allIndex;
    }
}

int Solver::getLocalSampleNum(){
	return numLocalSamples;
}

int Solver::getAllSampleNum(){
	return numAllSamples;
}

int Solver::getIterNum(){
	return iter;
}

//void Model::CheckSupportVector( Solver& solver, const SvmParameter& p){
void Model::CheckSupportVector_mpi(int maxid, double *alpha, SvmFeature *features, int *labels, int *index, int numSamples, const SvmParameter& p, double rho){
    maxId = maxid;
	param = p;
	SVvalues = 0;
	SVSamples = 0;
	numAllPos=0;
    numAllNeg=0;
    numLocalPos=0;
    numLocalNeg=0;
	this->rho = rho;
    //SV.boundSV = 0;
    int myrank;
    int commSz;
    //double *alpha = solver.getAlpha();
    //double *alpha = alpha;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	//MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::checkSV.Start();
	
	numProc = commSz;
    for(int k=0;k<numSamples;k++){
		if(alpha[k]!=0){
			if(labels[k]<0)
				numLocalNeg++;
			else
				numLocalPos++;
			
			//SVvalues+=index[k+1]-index[k];
			SVvalues+=maxId+1;
			SVSamples++;
		}
	}
	//cout<<"rank: "<<SVSamples<<endl;
	SVfeatures = new SvmFeature[SVvalues];
	SVlabels = new int[SVSamples];
	SVindex = new int[SVSamples];
	SValpha = new double[SVSamples];
	TimeProfile::checkSV.End();

	//MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::cmGather.Start();
	MPI_Reduce(&SVvalues,&allSVvalues,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&SVSamples,&allSVSamples,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&numLocalNeg,&numAllNeg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&numLocalPos,&numAllPos,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(myrank == 0){
		allSVfeatures = new SvmFeature[allSVvalues];
		allSVlabels = new int[allSVSamples];
		allSVindex = new int[allSVSamples];
		allSValpha = new double[allSVSamples];
		recValues = new int[numProc];
		v_displs = new int[numProc];
		recSamples = new int[numProc];
		s_displs = new int[numProc];
	}
	MPI_Gather(&SVSamples, 1, MPI_INT, recSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&SVvalues, 1, MPI_INT, recValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
	TimeProfile::cmGather.End();
	// MPI_Barrier(MPI_COMM_WORLD);

	TimeProfile::checkSV.Start();
	if(myrank == 0){
		for(int k=0;k<numProc;k++){
			if(k==0){	
				v_displs[k] = 0;
				s_displs[k] = 0;
			}
			else{
				v_displs[k] = v_displs[k-1] + recValues[k-1];
				s_displs[k] = s_displs[k-1] + recSamples[k-1];
			}
		}
	}
	int location = 0;
	int i = 0;
	for(int k=0;k<numSamples;k++){
		if(alpha[k]!=0){
			//memcpy(&SVfeatures[location],&features[index[k]],sizeof(SvmFeature)*(index[k+1]-index[k]));
			//location += index[k+1]-index[k];
			memcpy(&SVfeatures[location],&features[index[k]],sizeof(SvmFeature)*maxId);
			location += maxId+1;
			SVlabels[i] = labels[k];
			SValpha[i] = alpha[k];
			i++;
		}
	}
	TimeProfile::checkSV.End();
	if(i!=SVSamples)
		cout<<"wrong num of svsamples:"<<i<<" "<<SVSamples<<endl;
	if(location!=SVvalues)
		cout<<"wrong num of svvalue:"<<location<<" "<<SVvalues<<endl;

	// MPI_Barrier(MPI_COMM_WORLD);
	TimeProfile::cmGather.Start();
	MPI_Gatherv( SVfeatures, SVvalues, MPI_DOUBLE_INT, allSVfeatures, recValues, v_displs, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( SVlabels, SVSamples, MPI_INT, allSVlabels, recSamples, s_displs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( SValpha, SVSamples, MPI_DOUBLE, allSValpha, recSamples, s_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	TimeProfile::cmGather.End();
	// MPI_Barrier(MPI_COMM_WORLD);

	delete[] SVfeatures;
	delete[] SVlabels;
	delete[] SVindex;
	delete[] SValpha;
	if(myrank == 0){
		delete[] recValues;
		delete[] v_displs;
		delete[] recSamples;
		delete[] s_displs;
	}
	// int j = 0;
	// if(myrank == 0){
	// 	for(int k=0;k<allSVSamples;k++){
	// 		cout<<k<<" alpha:"<<allSValpha[k]<<" label:"<<allSVlabels[k];
	// 		while(allSVfeatures[j].id!=0){
	// 			cout<<" "<<allSVfeatures[j].id<<":"<<allSVfeatures[j].value;
	// 			j++;
	// 		}
	// 		cout<<endl;
	// 		j++;
	// 	}
	// }
    //cout<<SV.alpha.size()<<endl;
}

int Model::svm_save_model(const char *model_file_name)
{
	int myrank;
    int commSz;
    //double *alpha = solver.getAlpha();
    //double *alpha = alpha;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	//MPI_Barrier(MPI_COMM_WORLD);
	
	if(myrank == 0){
		TimeProfile::saveModel.Start();
		cout << endl<< "____________________________________" << endl;
        cout<<"Saving model..."<<endl;
        ofstream outfile(model_file_name);
		outfile << "kernel_type "<<param.kernelType<<endl; 
		outfile << "gamma "<<param.gamma<<endl; 
		outfile << "total_sv "<<allSVSamples<<endl; 
		outfile << "rho "<<rho<<endl; 
		outfile << "nr_sv "<<numAllPos<<" "<<numAllNeg<<endl; 
		outfile << "SV "<<endl; 
		int j = 0;
		for(int k=0;k<allSVSamples;k++){
			outfile << allSValpha[k]*allSVlabels[k]<<" "; 
			j = k*(maxId+1);
			while(allSVfeatures[j].id!=0){
				outfile<<allSVfeatures[j].id<<":"<<allSVfeatures[j].value<<" ";
				j++;
			}
			outfile<<endl;
			//j++;
		}
		outfile.close();
		TimeProfile::saveModel.End();
		cout<<"done"<<endl;;
		delete[] allSVfeatures;
		delete[] allSVlabels;
		delete[] allSVindex;
		delete[] allSValpha;
	}
	//MPI_Barrier(MPI_COMM_WORLD);
	

	return 0;
}

int Model::svm_load_model(const char *model_file_name){
	int myRank, commSz;
    allSVSamples = 0;
    //localStart = -1;
    //numAllPos = 0;
    //numAllNeg = 0;
    //numLocalNeg = 0;
    //numLocalPos = 0;
	allSVvalues = 0;
    double y_alpha;

    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    numProc = commSz;

	ifstream infile(model_file_name);
	if(myRank==0){
		cout << "------------------------------------" << endl;
    	cout<<"Reading modelfile...";
		if(!infile.is_open()){
        	cerr << "Error: Fail to open " << model_file_name << endl;
			MPI_Abort(MPI_COMM_WORLD,99);
        	//exit(1);
    	}
	}

    string str;
    int totalSV;
    int numLine=0;
    while(getline(infile, str)){
		//cout<<"#"<<endl;
        numLine++;
        stringstream strStream(str);
        string strData;
	    strStream>>strData;
		//cout<<strData<<endl;
        if(!strData.compare("kernel_type")){
			//cout<<"&"<<endl;
            strStream>>strData;
            param.kernelType = atoi(strData.c_str());
        }
        else if(!strData.compare("gamma")){
            strStream>>strData;
            param.gamma = atof(strData.c_str());
			//cout<<"gamma"<<param.gamma<<endl;
        }
        else if(!strData.compare("total_sv")){
            strStream>>strData;
            totalSV = atoi(strData.c_str());
        }
        else if(!strData.compare("rho")){
            strStream>>strData;
            rho = atof(strData.c_str());
        }
		else if(!strData.compare("nr_sv")){
            strStream>>strData;
            numAllPos = atoi(strData.c_str());
			strStream>>strData;
            numAllNeg = atoi(strData.c_str());
        }
        else if(!strData.compare("SV"))
            break;
		else if(numLine>6 && myRank == 0){
			cerr<<"Error: model file format is wrong!"<<endl;
			MPI_Abort(MPI_COMM_WORLD,99);
			//exit(1);
		}
    }
    //获取样本数据个数和element
    while(getline(infile, str)){
		//cout<<"$"<<endl;
        allSVSamples++;
        stringstream strStream(str);
        string strData;
		strStream>>strData;
		//cout<<strData<<endl;
        //getline(strStream, strData, ' ');
        //while(getline(strStream, strData, ' ')){
		while(strStream>>strData){
			//cout<<strData<<" t"<<endl;
            allSVvalues++;
        }
		//cout<<numAllSamples<<" "<<numValues<<endl;
    //system("pause");
	}
    if(myRank == 0 && (totalSV!=allSVSamples || numAllNeg+numAllPos!=allSVSamples)){
        cerr<<"Error: num of SV is wrong!"<<endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }
	//cout<<numLine<<endl;
    infile.clear();                                                                                                                                              
    infile.seekg(0, ios::beg);
    while(numLine!=0){
		//cout<<"%"<<endl;
        getline(infile, str);
        numLine--;
    }
	//cout<<numAllSamples<<" "<<numValues<<endl;
	//system("pause");
    SVyAlpha = new double[allSVSamples];
	allSVindex = new int[allSVSamples+1];
    allSVfeatures = new SvmFeature[allSVvalues+allSVSamples];
	norm = new double[allSVSamples];

    maxId = 0;
    int instMaxId;
    int j = 0;
    int i;
    for(i=0;i<allSVSamples;i++){
		//cout<<"^"<<endl;
        instMaxId = -1;
        getline(infile, str);
        allSVindex[i] = j;  //样本数据对于属性的索引

        stringstream strStream(str);
        string strData;
        //getline(strStream, strData, ' ');
		strStream>>strData;
        y_alpha = atof(strData.c_str());
        SVyAlpha[i] = y_alpha;
		norm[i] = 0;
        //cout<<SVyAlpha[i]<<"# "<<i<<" ";
        //while(getline(strStream, strData, ' ')){
		while(strStream>>strData){
            stringstream strStream2(strData);

            getline(strStream2, strData, ':');
            allSVfeatures[j].id = atoi(strData.c_str());

            instMaxId = allSVfeatures[j].id;
			//cout<<"instMaxId"<<instMaxId<<endl;
            getline(strStream2, strData);
            allSVfeatures[j].value = atof(strData.c_str());
            //cout<<allSVfeatures[j].id<<":"<<allSVfeatures[j].value<<"# ";
			norm[i]+=allSVfeatures[j].value*allSVfeatures[j].value;
            j++;
        }
		allSVfeatures[j].id = 0;
		allSVfeatures[j].value = 0;
		//cout<<allSVfeatures[j].id<<":"<<allSVfeatures[j].value<<"# ";
		j++;
		//allFeatures[j].id = 0;
        //cout<<endl;
        if(instMaxId > maxId){
			maxId = instMaxId;
		}
    }
	//cout<<"maxid"<<maxId<<endl;
    allSVindex[i]=allSVSamples+allSVvalues;	
	//cout<<i<<" "<<allIndex[i]<<" "<<numValues<<" "<<numAllSamples;
	infile.close();  
	if(myRank == 0)
		cout<<"done"<<endl;
    return 0;
}

void parseCommandLine(int argc, char **argv, SvmParameter *param,
	 char *dataFileName, char *modelFileName, char *timeFileName){

    int i;

    //参数默认值
    param->gamma = 1;
    param->hyperParmC = 1; 
    param->epsilon = 1e-3;
    param->kernelType = RBF;
    param->coef = 0;
    param->degree = 3;
    param->maxIter = 1000;
    param->commIter = 1000;
    param->cacheSize = 100;
	param->shrinking = 2;
	param->loadbalance = 0;

    //读命令行参数
    for( i = 1; i < argc; i++){
        if(argv[i][0] != '-')
            break;
        if(++i > argc-1){
            exit_with_help(0);
            // cerr << "The Command Line Param is wrong" << endl;
            // exit(1);
        }
        switch (argv[i-1][1]){
            case 'g':
                param->gamma = atof(argv[i]);
                break;
            case 'r':
                param->coef = atof(argv[i]);
                break;
            case 'd':
                param->degree = atoi(argv[i]);
                break;
            case 'c':
                param->hyperParmC = atof(argv[i]);
                break;
            case 'k':
                param->kernelType = atoi(argv[i]);
                break;
            case 'e':
                param->epsilon = atof(argv[i]);
                break;
            case 'i':
                param->commIter = atoi(argv[i]);
                break;
            case 'm':
                param->cacheSize = atof(argv[i]);
                break;
			case 's':
                param->shrinking = atoi(argv[i]);
                break;
			case 'l':
                param->loadbalance = atoi(argv[i]);
                break;
            default:
                cerr<<"Error: unknown option: -"<<argv[i-1][1]<<endl;
				exit_with_help(0);
                // cerr << "The Command Line Param is wrong" << endl;
                // exit(1);
        }
    }

    //数据文件名
    if(i < argc)    
        strcpy(dataFileName, argv[i]);
    else
        exit_with_help(0);
        
    //模型文件名
    // if(++i < argc)  
    //     strcpy(modelFileName, argv[i]);
    // else{
    char *p = strrchr(argv[i],'/');
	if(p==NULL)
		p = argv[i];
	else
		++p;
	sprintf(modelFileName,"%s.model",p);
	sprintf(timeFileName,"%s.time",p);
    //}

	// int myrank,commSz;
	// MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// MPI_Comm_size(MPI_COMM_WORLD, &commSz);
	// if(myrank == 0){
	// 	ofstream outfile(timeFileName);
	// 	outfile << "Num of cores: " << commSz<<endl;
	// 	outfile<<"The parameters of svm:	C:"<<param->hyperParmC<<" eps:"<<param->epsilon<<" gamma:"<<param->gamma<<endl;
	// 	outfile << "Train data filename: " << dataFileName<<endl;
	// 	outfile.close();
	// }
    
}

void checkParameter( const SvmParameter &param){
	
    // kernel_type, degree
	int kernelType = param.kernelType;
	if(kernelType < 0 || kernelType > 2){
		cerr << "Error: unknown kernel type" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }

	if(param.gamma < 0){
		cerr << "Error: gamma < 0" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }

	if(param.epsilon <= 0){
		cerr << "Error: eps <= 0" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }

    if(param.hyperParmC <= 0){
		cerr << "Error: C <= 0" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }

	if(param.cacheSize <= 0){
		cerr << "Error: cacheSize <= 0" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }

	if(param.shrinking != 0 && param.shrinking != 1 && param.shrinking != 2){
		cerr << "Error: -s(shrinking) param is 0 or 1" << endl;
		MPI_Abort(MPI_COMM_WORLD,99);
        //exit(1);
    }
	
}

void quick_sort(activeInfo *s, int l, int r){
    if (l < r){
        //Swap(s[l], s[(l + r) / 2]); //将中间的这个数和第一个数交换 参见注1
        int i = l, j = r;
		activeInfo x = s[l];
        while (i < j){
            while(i < j && s[j].active >= x.active) // 从右向左找第一个小于x的数
                j--;  
            if(i < j) 
                s[i++] = s[j];
            
            while(i < j && s[i].active < x.active) // 从左向右找第一个大于等于x的数
                i++;  
            if(i < j) 
                s[j--] = s[i];
        }
        s[i] = x;
        quick_sort(s, l, i - 1); // 递归调用 
        quick_sort(s, i + 1, r);
    }
}
