#include <cstdio>
#include <iostream>

#include "compute.hpp"


int main()
{
	float test[4] = {1, 2, 3, 4};
	float test2[4] = {5, 6, 7, 8};
	float ret[4];
	Compute c("add", CL_DEVICE_TYPE_GPU);

	c.set_buffer(test, 4);
	c.set_buffer(test2, 4);
	c.set_ret_buffer(ret, 4);
	c.run(4);

	for(auto &i: ret)
		std::cout << i << std::endl;
	system("pause");
}
