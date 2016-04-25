#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <ctime>
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>    


using namespace af;
using namespace std;

#include "compute.hpp"

//自定結構 每做完一次WeakLeaner 會產生一組WeakLeanerOutput的結構 再存到F二維陣列之中
struct WeakLeanerOutput
{
	public:
		float theta;
		float polarity;
		float final_error;

		WeakLeanerOutput()
		{
			theta = 0;
			polarity = 0;
			final_error = 0;
		}
};

class ReturnPair
{
	public:
		float previous_R_Sum;   //過去10、5、1分鐘的報酬率總合
		float target_R;			//未來30分鐘的報酬
		int pf_Index;			//正資料索引值
		int nf_Index;			//負資料索引值
};

bool CompareR(ReturnPair, ReturnPair);

WeakLeanerOutput weakLearn(float pf1[], float nf1[], float pw[], float nw[], int pf1_sn, int nf1_sn);
void AdaBoostTrain(float pf[][5000], float nf[][5000], int times);
void AdaBoostTest(float data_Fe[], float data_Re[]);
float MyRound(float number);

void KNN_Search(float test_Re[], ReturnPair pair_PF[], ReturnPair pair_NF[],
				float arr_Fe_PF[][233590], float arr_Fe_NF[][221629], float arr_Re_PF[][233590], float arr_Re_NF[][221629], 
				float real_Fe_PF[][5000], float real_Fe_NF[][5000], float real_Re_PF[][5000], float real_Re_NF[][5000]);

Compute compute("WeakLearn", CL_DEVICE_TYPE_GPU);

const int times = 500;	//訓練次數
float F[times][4];	//用二維矩陣 存放每次訓練完之結果 4分別代表著 1. selectif(選到的Feature) 2. polarity(右邊是正or負資料) 3.theta 4. alpha值 
const int Train_PF_Num = 233590;  // positive number Traing Data
const int Train_NF_Num = 221629;  // negative number Traing Data
const int KNN_ForTrainData = 5000; //例如:KNN找出一萬筆 正負訓練資料各五千
const int Test_PF_Num = 5;
const int Test_NF_Num = 1;
//const int Test_PF_Num = 18341;	// positive number Testing Data
//const int Test_NF_Num = 18219;  // negative number Testing Data
const int fn = 162;		// feature number
const int rn = 4;		// return number

float success_Count = 0.0;
float fail_Count = 0.0;
float total_Count = 0.0;
float predict_Rate = 0.0;
float total_Profit = 0.0;

void ArrayPrint(float arr[])
{
	cout << arr[3];
};

int main()
{
	//auto arr = new float[2][4];
	//arr[0][0] = 0;
	//arr[0][1] = 1;
	//arr[0][2] = 2;
	//arr[0][3] = 3;
	//arr[1][0] = 4;
	//arr[1][1] = 5;
	//arr[1][2] = 6;
	//arr[1][3] = 7;
	//ArrayPrint(arr[][1]);
	//system("pause");

	//float *output = weakLearn(pf[0], nf[0], pw[0], nw[0], sizeof(pf) / sizeof(pf[0]), sizeof(nf) / sizeof(nf[0]));

	//讀資料是全部都讀 共分兩大類Feautre與Return
	char file_Train_Fe_PF[] = "G:\\2001-2012_F_Train_PF.txt";   //233590*162
	char file_Train_Fe_NF[] = "G:\\2001-2012_F_Train_NF.txt";   //221629*162
	char file_Test_Fe_PF[] = "G:\\2013_F_Test_PF.txt";	   //18341*162
	char file_Test_Fe_NF[] = "G:\\2013_F_Test_NF.txt";	   //18219*162

	char file_Train_Re_PF[] = "G:\\2001-2012_Re_Train_PF.txt";   //233590*4
	char file_Train_Re_NF[] = "G:\\2001-2012_Re_Train_NF.txt";   //221629*4
	char file_Test_Re_PF[] = "G:\\2013_Re_Test_PF.txt";	   //18341*4
	char file_Test_Re_NF[] = "G:\\2013_Re_Test_NF.txt";	   //18219*4 

	//最原始的Traing Data 數量龐大
	auto arr_Train_Fe_PF = new float[fn][Train_PF_Num];
	auto arr_Train_Fe_NF = new float[fn][Train_NF_Num];
	auto arr_Train_Re_PF = new float[rn][Train_PF_Num];
	auto arr_Train_Re_NF = new float[rn][Train_NF_Num];

	//真正去學習的只有從KNN找出的一萬筆 正負各給五千
	auto real_Train_Fe_PF = new float[fn][KNN_ForTrainData];
	auto real_Train_Fe_NF = new float[fn][KNN_ForTrainData];
	auto real_Train_Re_PF = new float[rn][KNN_ForTrainData];
	auto real_Train_Re_NF = new float[rn][KNN_ForTrainData];

	auto arr_Test_Fe_PF = new float[fn][Test_PF_Num];
	auto arr_Test_Fe_NF = new float[fn][Test_NF_Num];
	auto arr_Test_Re_PF = new float[rn][Test_PF_Num];
	auto arr_Test_Re_NF = new float[rn][Test_NF_Num];

	//KNN Search Function中所需要的資料結構 
	//將(1)過去報酬率 (2)未來報酬率 (3)索引值 綁定後再做排序
	auto pair_PF = new ReturnPair[Train_PF_Num];
	auto pair_NF = new ReturnPair[Train_NF_Num];

	/*
	利用'\t'當作分隔符號
	將[rows][feature]轉置成[feature][rows]方便平行運算的Code處理
	有Feature以及Return兩種資料需要處理
	*/

	fstream fp1;
	char line1[256];

	fp1.open(file_Train_Fe_PF, ios::in);//開啟檔案
	if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Fe_PF << endl;
	}

	int i1 = 0;
	int j1 = 0;

	while (fp1.getline(line1, sizeof(line1), '\t'))
	{
		if (i1 == Train_PF_Num)
			break;

		//atof是一種將字串轉為浮點數的函數
		arr_Train_Fe_PF[j1][i1] = atof(line1);

		j1++;

		if (j1 == fn)
		{
			j1 = 0;
			i1++;
		}
	}
	fp1.close();//關閉檔案

	fp1.open(file_Train_Re_PF, ios::in);//開啟檔案
	if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Re_PF << endl;
	}

	i1 = 0;
	j1 = 0;

	while (fp1.getline(line1, sizeof(line1), '\t'))
	{
		if (i1 == Train_PF_Num)
			break;

		//atof是一種將字串轉為浮點數的函數
		arr_Train_Re_PF[j1][i1] = atof(line1);

		j1++;

		if (j1 == rn)
		{
			j1 = 0;
			i1++;
		}
	}
	fp1.close();//關閉檔案



	fstream fp2;
	char line2[256];

	fp2.open(file_Train_Fe_NF, ios::in);//開啟檔案
	if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Fe_NF << endl;
	}

	int i2 = 0;
	int j2 = 0;

	while (fp2.getline(line2, sizeof(line2), '\t'))
	{
		if (i2 == Train_NF_Num)
			break;

		arr_Train_Fe_NF[j2][i2] = atof(line2);

		j2++;

		if (j2 == fn)
		{
			j2 = 0;
			i2++;
		}
	}

	fp2.close();//關閉檔案

	fp2.open(file_Train_Re_NF, ios::in);//開啟檔案
	if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Re_NF << endl;
	}

	i2 = 0;
	j2 = 0;

	while (fp2.getline(line2, sizeof(line2), '\t'))
	{
		if (i2 == Train_NF_Num)
			break;

		arr_Train_Re_NF[j2][i2] = atof(line2);

		j2++;

		if (j2 == rn)
		{
			j2 = 0;
			i2++;
		}
	}

	fp2.close();//關閉檔案
	

	fstream fp3;
	char line3[256];

	fp3.open(file_Test_Fe_PF, ios::in);//開啟檔案
	if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Fe_PF << endl;
	}

	int i3 = 0;
	int j3 = 0;

	while (fp3.getline(line3, sizeof(line3), '\t'))
	{
		if (i3 == Test_PF_Num)
			break;

		arr_Test_Fe_PF[j3][i3] = atof(line3);

		j3++;

		if (j3 == fn)
		{
			j3 = 0;
			i3++;
		}
	}

	fp3.close();//關閉檔案

	fp3.open(file_Test_Re_PF, ios::in);//開啟檔案
	if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Re_PF << endl;
	}

	i3 = 0;
	j3 = 0;

	while (fp3.getline(line3, sizeof(line3), '\t'))
	{
		if (i3 == Test_PF_Num)
			break;

		arr_Test_Re_PF[j3][i3] = atof(line3);

		j3++;

		if (j3 == rn)
		{
			j3 = 0;
			i3++;
		}
	}

	fp3.close();//關閉檔案

	fstream fp4;
	char line4[256];

	fp4.open(file_Test_Fe_NF, ios::in);//開啟檔案
	if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Fe_NF << endl;
	}

	int i4 = 0;
	int j4 = 0;

	while (fp4.getline(line4, sizeof(line4), '\t'))
	{
		if (i4 == Test_NF_Num)
			break;

		arr_Test_Fe_NF[j4][i4] = atof(line4);

		j4++;

		if (j4 == fn)
		{
			j4 = 0;
			i4++;
		}
	}

	fp4.close();//關閉檔案

	fp4.open(file_Test_Re_NF, ios::in);//開啟檔案
	if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Re_NF << endl;
	}

	i4 = 0;
	j4 = 0;

	while (fp4.getline(line4, sizeof(line4), '\t'))
	{
		if (i4 == Test_NF_Num)
			break;

		arr_Test_Re_NF[j4][i4] = atof(line4);

		j4++;

		if (j4 == rn)
		{
			j4 = 0;
			i4++;
		}
	}

	fp4.close();//關閉檔案

	for (size_t i = 0; i < Test_PF_Num+Test_NF_Num; i++)
	{
		cout << "i= " << i << "\n";

		//從原始巨量的資料中 依照歐幾里得距離 找出最相似的 各五千筆漲跌資料
		
		clock_t begin = clock();

		auto return_Test = new float[rn];
		auto feature_Test = new float[fn];

		//postive
		if (i < Test_PF_Num)
		{		
			for (size_t a = 0; a < rn; a++)
			{
				return_Test[a] = arr_Test_Re_PF[a][i];
			}
			for (size_t b = 0; b < fn; b++)
			{
				feature_Test[b] = arr_Test_Fe_PF[b][i];
			}
		}
		//negative
		else if (i < Test_NF_Num)
		{
			for (size_t a = 0; a < rn; a++)
			{
				return_Test[a] = arr_Test_Re_NF[a][i];
			}
			for (size_t b = 0; b < fn; b++)
			{
				feature_Test[b] = arr_Test_Fe_NF[b][i];
			}
		}

		KNN_Search(return_Test, pair_PF, pair_NF,
			arr_Train_Fe_PF, arr_Train_Fe_NF, arr_Train_Re_PF, arr_Train_Re_NF,
			real_Train_Fe_PF, real_Train_Fe_NF, real_Train_Re_PF, real_Train_Re_NF);

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "One KNN Time: " << elapsed_secs << " seconds!!!!" << endl;
		system("pause");

		clock_t train_beginTime = clock();
		AdaBoostTrain(real_Train_Fe_PF, real_Train_Fe_NF, times);
		clock_t train_endTime = clock();
		double train_Sec = double(train_endTime - train_beginTime) / CLOCKS_PER_SEC;
		cout << "Adaboost Train: " << train_Sec << " seconds!!!!" << endl;
		system("pause");

		AdaBoostTest(feature_Test, return_Test);
		system("pause");
	}

	cout << "準確率= " << success_Count / total_Count << "\n";
	cout << "總獲利= " << total_Profit << "\n";
	system("pause");


	//釋放記憶體
	for (size_t f = 0; f < fn; f++)
	{
		delete[] arr_Train_Fe_PF[f];
		delete[] arr_Train_Fe_NF[f];
		delete[] real_Train_Fe_PF[f];
		delete[] real_Train_Fe_NF[f];
		delete[] arr_Test_Fe_PF[f];
		delete[] arr_Test_Fe_NF[f];
	}

	for (size_t r = 0; r < rn; r++)
	{
		delete[] arr_Train_Re_PF[r];
		delete[] arr_Train_Re_NF[r];
		delete[] arr_Train_Re_PF[r];
		delete[] arr_Train_Re_NF[r];
		delete[] arr_Test_Re_PF[r];
		delete[] arr_Test_Re_NF[r];
	}

	delete[] arr_Train_Fe_PF;
	delete[] arr_Train_Fe_NF;
	delete[] real_Train_Fe_PF;
	delete[] real_Train_Fe_NF;
	delete[] arr_Test_Fe_PF;
	delete[] arr_Test_Fe_NF;
	delete[] arr_Train_Re_PF;
	delete[] arr_Train_Re_NF;
	delete[] arr_Train_Re_PF;
	delete[] arr_Train_Re_NF;
	delete[] arr_Test_Re_PF;
	delete[] arr_Test_Re_NF;

	delete[] pair_PF;
	delete[] pair_NF;

	system("pause");
}

void AdaBoostTrain(float pf[][5000], float nf[][5000], int times)
{
	float *pw = new float[KNN_ForTrainData];
	float *nw = new float[KNN_ForTrainData];

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		pw[i] = 0.5 / KNN_ForTrainData;
	}

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		nw[i] = 0.5 / KNN_ForTrainData;
	}

	float wsum = 0;

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		wsum = wsum + pw[i];
	}
	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		wsum = wsum + nw[i];
	}

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		pw[i] /= wsum;
	}

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		nw[i] /= wsum;
	}

	float ret[fn][3];

	//OpenCL
	int pf_shape[2] = { fn, KNN_ForTrainData };
	int nf_shape[2] = { fn, KNN_ForTrainData };

	compute.set_buffer((float *)pf, fn * KNN_ForTrainData*sizeof(float));
	compute.set_buffer((float *)nf, fn * KNN_ForTrainData*sizeof(float));

	compute.set_buffer((float *)pw, KNN_ForTrainData*sizeof(float));
	compute.set_buffer((float *)nw, KNN_ForTrainData*sizeof(float));

	compute.set_buffer((int *)pf_shape, 2 * sizeof(int));
	compute.set_buffer((int *)nf_shape, 2 * sizeof(int));

	//compute.set_buffer(1);

	compute.set_ret_buffer((float *)ret, fn * 3 * sizeof(float));

	for (int i = 0; i < times; i++)
	{
		float wsum = 0;

		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			wsum = wsum + pw[i];
		}
		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			wsum = wsum + nw[i];
		}

		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			pw[i] /= wsum;
		}
		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			nw[i] /= wsum;
		}

		WeakLeanerOutput *output = new WeakLeanerOutput[fn];
		
		compute.reset_buffer(2, pw);
		compute.reset_buffer(3, nw);

		//幾個Kernel在跑
		compute.run(fn);

		for (int y = 0; y < fn; y++)
		{
			output[y].final_error = ret[y][0];
			output[y].polarity = ret[y][1];
			output[y].theta = ret[y][2];
		}

		float error = 1, theta = 0, polarity = 1;
		int selectif = -1;
		float beta;

		for (int k = 0; k < fn; k++)
		{
			if (output[k].final_error < error)
			{
				error = output[k].final_error;
				theta = output[k].theta;
				polarity = output[k].polarity;
				selectif = k;
			}
		}

		//if (error > output[i].final_error)
		//{
		//	error = output[i].final_error;
		//	polarity = output[i].polarity;
		//	theta = output[i].theta;
		//	selectif = i;	//最好的那一"行"特徵 (從0開始算，跟Matlab誤差1)
		//}

		//printf("%f, %f, %f, %f\n", output[i].theta, output[i].polarity, output[i].final_error, selectif);
		

		beta = error / (1 - error);

		if (polarity == 1)
		{
			for (int i = 0; i < KNN_ForTrainData; i++)
			{
				if (pf[selectif][i] >= theta)
				{
					pw[i] = pw[i] * beta;
				}
			}

			for (int i = 0; i < KNN_ForTrainData; i++)
			{
				if (nf[selectif][i] < theta)
				{
					nw[i] = nw[i] * beta;
				}
			}
		}
		else
		{
			for (int i = 0; i < KNN_ForTrainData; i++)
			{
				if (pf[selectif][i] < theta)
				{
					pw[i] = pw[i] * beta;
				}
			}
			for (int i = 0; i < KNN_ForTrainData; i++)
			{
				if (nf[selectif][i] >= theta)
				{
					nw[i] = nw[i] * beta;
				}
			}
		}

		//printf("i=%d\n", i);

		F[i][0] = selectif;
		F[i][1] = polarity;
		F[i][2] = theta;
		F[i][3] = log(1 / beta);

		//把alpha值 四捨五入至小數第四位
		//F[i][3] = MyRound(F[i][3]);

		//秀出每次訓練完 最好的那一刀　500次就有500組
		//printf("[%d] , %f, %f, %f, %f\n", i + 1, F[i][0] + 1, F[i][1], F[i][2], F[i][3]);

		//if (i % 100 == 0)
		//	system("pause");

	}

	//釋放記憶體
	delete[] pw;
	delete[] nw;
}


//自製小數點四捨五入至小數第四位
float MyRound(float number)
{
	float f = floor(number * 10000 + 0.5) / 10000;
	return f;
}

/*
	一分鐘建一個模型，測也只測這一分鐘
	只要最後alpha>=0.5就是好
*/
void AdaBoostTest(float data_Fe[], float data_Re[])
{
	float alphaSum = 0;
	float score = 0;

	for (size_t i = 0; i < times; i++)
	{
		alphaSum += F[i][3];

		//Polarity為正
		if (F[i][1] == 1)
		{
			if (data_Fe[(int)F[i][0]] >= F[i][2])
				score += F[i][3];
		}
		//Polarity為负 
		else
		{
			if (data_Fe[(int)F[i][0]] < F[i][2])
				score += F[i][3];
		}	
	}

	float finalDecision = score / alphaSum;

	cout << "\nsocre= " << score;
	cout << "\nalphaSum = " << alphaSum;
	cout << "\nfinalDecision= " << finalDecision;

	//看多
	if (finalDecision >= 0.5)
	{
		if (data_Re[3] >= 0)
		{
			success_Count++;
			total_Profit += data_Re[3];
		}
		else
		{
			total_Profit -= abs(data_Re[3]);
			fail_Count++;
		}
	}
	//看空
	else
	{
		if (data_Re[3] < 0)
		{
			success_Count++;
			total_Profit += abs(data_Re[3]);
		}
		else
		{
			total_Profit -= abs(data_Re[3]);
			fail_Count++;
		}
	}

	total_Count++;

}

//根據原始資料 尋找相似的10,000筆資料出來建模 (正負資料各五千)
//傳入的資料結構 列=Feature, 欄=分鐘資料
void KNN_Search(float test_Re[], ReturnPair pair_PF[], ReturnPair pair_NF[],
				float arr_Fe_PF[][233590], float arr_Fe_NF[][221629], float arr_Re_PF[][233590], float arr_Re_NF[][221629], 
				float real_Fe_PF[][5000], float real_Fe_NF[][5000], float real_Re_PF[][5000], float real_Re_NF[][5000])
{
	// p means previous, t means target, 測試的這一分鐘之四個Return宣告如下
	float return_p10 = test_Re[0];
	float return_p5 = test_Re[1];
	float return_p1 = test_Re[2];
	float return_t30 = test_Re[3];

	////cout << "\n" << return_p10 << ", " << return_p5 << ", " << return_p1 << ", " << return_t30 << "\n";
	////system("pause");

	int zero = 0;

	//從所有歷史資料中，根據現在Testing這筆資料之報酬率，計算距離總合與我最近的(最像的)
	for (size_t i = 0; i < Train_PF_Num; i++)
	{
		pair_PF[i].previous_R_Sum = abs(return_p10 - arr_Re_PF[0][i]) 
									  + abs(return_p5 - arr_Re_PF[1][i]) 
									  + abs(return_p1 - arr_Re_PF[2][i]);
		pair_PF[i].target_R = arr_Re_PF[3][i];
		pair_PF[i].pf_Index = i;

		//cout << pair_PF[i].previous_R_Sum << "\n";
		//cout << pair_PF[i].target_R << "\n";
		//system("pause");
	}

	//從所有歷史資料中，根據現在Testing這筆資料之報酬率，計算距離總合與我最近的(最像的)
	for (size_t j = 0; j < Train_NF_Num; j++)
	{	
		pair_NF[j].previous_R_Sum = abs(return_p10 - arr_Re_NF[0][j])
								   + abs(return_p5 - arr_Re_NF[1][j]) 
								   + abs(return_p1 - arr_Re_NF[2][j]);

		pair_NF[j].target_R = arr_Re_NF[3][j];
		pair_NF[j].nf_Index = j;
	}

	//檢查排序前的內容內容
	//cout << "UnSorted: " << "\n";
	//for (size_t i = 0; i < 30000; i++)
	//	cout << pair_PF[i].previous_R_Sum << "\n";

	//自定義排序方法
	sort(pair_PF, pair_PF + Train_PF_Num, CompareR);
	sort(pair_NF, pair_NF + Train_NF_Num, CompareR);

	//排序後 就可以抓出正負最像的那五千筆
	for (size_t i = 0; i < KNN_ForTrainData; i++)
	{
		int pf_Idx = pair_PF[i].pf_Index;
		int nf_Idx = pair_NF[i].nf_Index;

		//cout << "Feature PF NF \n";

		//找出相對應的特徵
		for (size_t f = 0; f < fn; f++)
		{
			real_Fe_PF[f][i] = arr_Fe_PF[f][pf_Idx];
			real_Fe_NF[f][i] = arr_Fe_NF[f][nf_Idx];

			//cout << real_Fe_PF[f][i] << "\n";
			//cout << real_Fe_NF[f][i] << "\n";
		}

		//cout << "Return PF NF \n";

		//找出相對應的報酬率
		for (size_t r = 0; r < rn; r++)
		{
			real_Re_PF[r][i] = arr_Re_PF[r][pf_Idx];
			real_Re_NF[r][i] = arr_Re_NF[r][nf_Idx];

			//cout << real_Re_PF[r][i] << "\n";
			//cout << real_Re_NF[r][i] << "\n";
		}

		//cout << "PF\n";
		//cout << "Distance Sum " << pair_PF[i].previous_R_Sum << "\n";
		//cout << "Target " << pair_PF[i].target_R << "\n";
		//cout << "Index " << pair_PF[i].pf_Index << "\n";

		//cout << "NF\n";
		//cout << "Distance Sum " << pair_NF[i].previous_R_Sum << "\n";
		//cout << "Target " << pair_NF[i].target_R << "\n";
		//cout << "Index " << pair_NF[i].nf_Index << "\n";

		//if (i % 100 == 0)
		//	system("pause");

		//if (pair_PF[i].previous_R_Sum == 0)
		//{
		//	cout << " =0 " << "\n";
		//}
		//else if (pair_PF[i].previous_R_Sum < 0)
		//{
		//	cout << " <0 " << "\n";
		//}
		//else
		//{
		//	cout << " >0 " << "\n";
		//}
		
		//printf("%f", pair_PF[i].previous_R_Sum);
		//system("pause");
	}
	//system("pause");
}

bool CompareR(ReturnPair r1 , ReturnPair r2)
{
	return r1.previous_R_Sum < r2.previous_R_Sum;	
}
