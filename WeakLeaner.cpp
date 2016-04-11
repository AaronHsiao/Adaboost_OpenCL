#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <ctime>
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

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

WeakLeanerOutput weakLearn(float pf1[], float nf1[], float pw[], float nw[], int pf1_sn, int nf1_sn);
void AdaBoostTrain(float pf[][2429], float nf[][4548], int times);
void AdaBoostTest(float data[][472]);
float MyRound(float number);

Compute compute("WeakLearn", CL_DEVICE_TYPE_GPU);

const int times = 500;	//訓練次數
float F[times][4];	//用二維矩陣 存放每次訓練完之結果 4分別代表著 1. selectif(選到的Feature) 2. polarity(右邊是正or負資料) 3.theta 4. alpha值 
const int Train_PF_Num = 2429;  // positive number Traing Data
const int Train_NF_Num = 4548;  // negative number Traing Data
const int Test_PF_Num = 472;	// positive number Testing Data
const int Test_NF_Num = 23573;  // negative number Testing Data
const int fn = 2101;		// feature number

int main()
{
	//float *output = weakLearn(pf[0], nf[0], pw[0], nw[0], sizeof(pf) / sizeof(pf[0]), sizeof(nf) / sizeof(nf[0]));

	char file_Train_PF1[] = "G:\\Train_PF1.txt";   //2429*2101
	char file_Train_NF1[] = "G:\\Train_NF1.txt";   //4548*2101
	char file_Test_PF1[] = "G:\\Test_PF1.txt";	   //472*2101
	char file_Test_NF1[] = "G:\\Test_NF1.txt";	   //23573*2101

	auto arr_Train_PF1 = new float[fn][Train_PF_Num];
	fstream fp1;
	char line1[256];

	fp1.open(file_Train_PF1, ios::in);//開啟檔案
	if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_PF1 << endl;
	}

	int i1 = 0;
	int j1 = 0;

	while (fp1.getline(line1, sizeof(line1), '\t'))
	{
		if (i1 == Train_PF_Num)
			break;

		arr_Train_PF1[j1][i1] = atof(line1);

		j1++;

		if (j1 == fn)
		{
			j1 = 0;
			i1++;
		}
	}

	fp1.close();//關閉檔案


	auto arr_Train_NF1 = new float[fn][Train_NF_Num];
	fstream fp2;
	char line2[256];

	fp2.open(file_Train_NF1, ios::in);//開啟檔案
	if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_NF1 << endl;
	}

	int i2 = 0;
	int j2 = 0;

	while (fp2.getline(line2, sizeof(line2), '\t'))
	{
		if (i2 == Train_NF_Num)
			break;

		arr_Train_NF1[j2][i2] = atof(line2);

		j2++;

		if (j2 == fn)
		{
			j2 = 0;
			i2++;
		}
	}

	fp2.close();//關閉檔案

	
	auto arr_Test_PF1 = new float[fn][Test_PF_Num];

	//動態配置二維矩陣 否則會StackOverFlow
	//arr_Test_PF1 = new float*[row_Test_PF1];
	//for (int i = 0; i<row_Test_PF1; i++)
	//	arr_Test_PF1[i] = new float[Test_PF_Num];

	fstream fp3;
	char line3[256];

	fp3.open(file_Test_PF1, ios::in);//開啟檔案
	if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_PF1 << endl;
	}

	int i3 = 0;
	int j3 = 0;

	while (fp3.getline(line3, sizeof(line3), '\t'))
	{
		if (i3 == Test_PF_Num)
			break;

		arr_Test_PF1[j3][i3] = atof(line3);

		j3++;

		if (j3 == fn)
		{
			j3 = 0;
			i3++;
		}
	}

	fp3.close();//關閉檔案


	auto arr_Test_NF1 = new float[fn][Test_NF_Num];

	////動態配置二維矩陣 否則會StackOverFlow
	//arr_Test_NF1 = new float*[row_Test_NF1];
	//for (int i = 0; i<row_Test_NF1; i++)
	//	arr_Test_NF1[i] = new float[column_Test_NF1];

	fstream fp4;
	char line4[256];

	fp4.open(file_Test_NF1, ios::in);//開啟檔案
	if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_NF1 << endl;
	}

	int i4 = 0;
	int j4 = 0;

	while (fp4.getline(line4, sizeof(line4), '\t'))
	{
		if (i4 == Test_NF_Num)
			break;

		arr_Test_NF1[j4][i4] = atof(line4);

		j4++;

		if (j4 == fn)
		{
			j4 = 0;
			i4++;
		}
	}

	fp4.close();//關閉檔案

	clock_t begin = clock();
	
	AdaBoostTrain(arr_Train_PF1, arr_Train_NF1, times);
	//int TP = AdaBoostTest(arr_Test_PF1[0], row_Test_PF1, column_Test_NF1);
	//printf("%f/n", TP / row_Test_PF1);
	//int FP = row_Test_NF1-AdaBoostTest(arr_Test_NF1[0], row_Test_NF1, column_Test_NF1);
	//printf("%f/n", FP / row_Test_NF1);
	AdaBoostTest(arr_Test_PF1);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Time: " << elapsed_secs << " seconds!!!!" << endl;

	////釋放記憶體
	//for (int i = 0; i < row_Train_PF1; i++)
	//{
	//	delete[] arr_Train_PF1[i];
	//}

	delete[] arr_Train_PF1;
	delete[] arr_Train_NF1;

	system("pause");
}

void AdaBoostTrain(float pf[][2429], float nf[][4548], int times)
{
	float *pw = new float[Train_PF_Num];
	float *nw = new float[Train_NF_Num];

	for (int i = 0; i < Train_PF_Num; i++)
	{
		pw[i] = 0.5 / Train_PF_Num;
	}

	for (int i = 0; i < Train_NF_Num; i++)
	{
		nw[i] = 0.5 / Train_NF_Num;
	}

	float wsum = 0;

	for (int i = 0; i < Train_PF_Num; i++)
	{
		wsum = wsum + pw[i];
	}
	for (int i = 0; i < Train_NF_Num; i++)
	{
		wsum = wsum + nw[i];
	}

	for (int i = 0; i < Train_PF_Num; i++)
	{
		pw[i] /= wsum;
	}

	for (int i = 0; i < Train_NF_Num; i++)
	{
		nw[i] /= wsum;
	}

	float ret[2101][3];

	//OpenCL
	int pf_shape[2] = { fn, Train_PF_Num };
	int nf_shape[2] = { fn, Train_NF_Num };

	compute.set_buffer((float *)pf, fn * Train_PF_Num*sizeof(float));
	compute.set_buffer((float *)nf, fn * Train_NF_Num*sizeof(float));

	compute.set_buffer((float *)pw, Train_PF_Num*sizeof(float));
	compute.set_buffer((float *)nw, Train_NF_Num*sizeof(float));

	compute.set_buffer((int *)pf_shape, 2 * sizeof(int));
	compute.set_buffer((int *)nf_shape, 2 * sizeof(int));

	//compute.set_buffer(1);

	compute.set_ret_buffer((float *)ret, fn * 3 * sizeof(float));

	for (int i = 0; i < times; i++)
	{
		float wsum = 0;

		for (int i = 0; i < Train_PF_Num; i++)
		{
			wsum = wsum + pw[i];
		}
		for (int i = 0; i < Train_NF_Num; i++)
		{
			wsum = wsum + nw[i];
		}

		for (int i = 0; i < Train_PF_Num; i++)
		{
			pw[i] /= wsum;
		}
		for (int i = 0; i < Train_NF_Num; i++)
		{
			nw[i] /= wsum;
		}

		WeakLeanerOutput *output = new WeakLeanerOutput[fn];
		

		compute.reset_buffer(2, pw);
		compute.reset_buffer(3, nw);


		//幾個Kernel在跑
		compute.run(2101);

	
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
			for (int i = 0; i < Train_PF_Num; i++)
			{
				if (pf[selectif][i] >= theta)
				{
					pw[i] = pw[i] * beta;
				}
			}

			for (int i = 0; i < Train_NF_Num; i++)
			{
				if (nf[selectif][i] < theta)
				{
					nw[i] = nw[i] * beta;
				}
			}
		}
		else
		{
			for (int i = 0; i < Train_PF_Num; i++)
			{
				if (pf[selectif][i] < theta)
				{
					pw[i] = pw[i] * beta;
				}
			}
			for (int i = 0; i < Train_NF_Num; i++)
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
		F[i][3] = MyRound(F[i][3]);

		/*if (i % 100 == 0)*/
			printf("[%d] , %f, %f, %f, %f\n", i + 1, F[i][0] + 1, F[i][1], F[i][2], F[i][3]);

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

void AdaBoostTest(float data[][472])
{
	int F_Length = sizeof(F) / sizeof(F[0]);
	float alphaSum = 0;
	float *score = new float[Test_PF_Num];

	for (size_t a = 0; a < Test_PF_Num; a++)
		score[a] = 0.0;


	//cout << "\n\nF length= " << F_Length;
	//cout << "\alphaSum= " << alphaSum;
	//system("pause");

	for (size_t i = 0; i < F_Length; i++)
	{
		alphaSum += F[i][3];

		for (size_t j = 0; j < Test_PF_Num; j++)
		{
			//Polarity為正
			if (F[i][1] == 1)
			{
				if (data[(int)F[i][0]][j] >= F[i][2])
					score[j] += F[i][3];
			}
			//Polarity為负 
			else
			{
				if (data[(int)F[i][0]][j] < F[i][2])
					score[j] += F[i][3];
			}
		}
	}

	cout << "\nalphaSum = " << alphaSum << "\n";

	float scoreSum = 0;
	float greatScore = 0.0;

	for (size_t z = 0; z < Test_PF_Num; z++)
	{
		scoreSum += score[z];
		cout << z << ", " << score[z] / alphaSum << "\n";
		
		if (score[z] / alphaSum > 0.5)
			greatScore += 1;
		//if (z % 100 == 0)
		//	system("pause");
	}
		
	cout << "\ngreatSocre= " << greatScore;
	cout << "\npercentRate= " << greatScore / Test_PF_Num;

	//cout << "\nscore = " << scoreSum / (alphaSum*Test_PF_Num) << "\n";

	//for (size_t z = 0; z < Test_PF_Num; z++)
	//	cout << "socre[" << z << "] = " << score[z] << "  score/alphaSum= " << score[z] / alphaSum <<"\n";

	//system("pause");

}
