#include <cstdio>
#include <iostream>

#include "compute.hpp"

using namespace std;


//output[] = {error, polarity, theta}
void weakLearn(float pf1[], float nf1[], float pw[], float nw[], int pf1_sn, int nf1_sn)
{
	float max = 0, min = 2000000, error = 1, theta = 0, polarity = 1;
	//找最大 最小值
	for (int i = 0; i < pf1_sn; i++)
	{
		if (pf1[i] > max)
		{
			max = pf1[i];
		}
		if (pf1[i] < min)
		{
			min = pf1[i];
		}
	}
	for (int i = 0; i < nf1_sn; i++)
	{
		if (nf1[i] > max)
		{
			max = nf1[i];
		}
		if (nf1[i] < min)
		{
			min = nf1[i];
		}
	}

	//找最好的一刀
	for (int j = 1; j < 10; j++)
	{
		float theta1 = (max - min) / 10 * j + min;
		float error1 = 0;
		float polarity1 = 1;
		for (int i = 0; i < pf1_sn; i++)
		{
			if (pf1[i] < theta1)
			{
				error1 = error1 + pw[i];
			}
		}
		for (int i = 0; i < nf1_sn; i++)
		{
			if (nf1[i] > theta1)
			{
				error1 = error1 + nw[i];
			}
		}
		if (error1 > 0.5)
		{
			polarity1 = -1;
			error1 = 1 - error1;
		}
		if (error1 < error)
		{
			error = error1;
			polarity = polarity1;
			theta = theta1;
		}
	}

	cout << error << endl;
	cout << polarity << endl;
	cout << theta << endl;
	

}

int main()
{
	float pf[1][4] = {
		{ 2500, 1890, 1904, 2300 }
	};
	float pw[1][4] = {
		{ 0.1, 0.1, 0.1, 0.1 }
	};
	float nf[1][6] = {
		{ 1142, 2080, 15204, 2115, 1170, 6115 }
	};
	float nw[1][6] = {
		{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }
	};
	int pf_shape[2] = { 1, 4 };
	int nf_shape[2] = { 1, 6 };
	float ret[1][3];

	Compute c("WeakLearn", CL_DEVICE_TYPE_GPU);

	c.set_buffer((float *)pf, 1 * 4 * sizeof(float));
	c.set_buffer((float *)nf, 1 * 6 * sizeof(float));

	c.set_buffer((float *)pw, 1 * 4 * sizeof(float));
	c.set_buffer((float *)nw, 1 * 6 * sizeof(float));

	c.set_buffer((int *)pf_shape, 2 * sizeof(int));
	c.set_buffer((int *)nf_shape, 2 * sizeof(int));

	c.set_ret_buffer((float *)ret, 1 * 3 * sizeof(float));

	c.run(1, 4);

	std::cout << ret[0][0] << std::endl;
	std::cout << ret[0][1] << std::endl;
	std::cout << ret[0][2] << std::endl;

	c.reset_buffer();  // obj `c` can reuse for next `set_buffer` and `run`
	system("pause");
	weakLearn(pf[0], nf[0], pw[0], nw[0], 4, 6);
	system("pause");
}

