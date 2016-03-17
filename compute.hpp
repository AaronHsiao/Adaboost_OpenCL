#include <CL/cl.hpp>

#include <cassert>
#include <iostream>
#include <fstream>

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef COMPUTE_HPP  // include guard
#define COMPUTE_HPP


class Compute
{
public:
	// Platform
	std::vector< cl::Platform > platforms;
	cl::Platform platform;
	// Device
	std::vector< cl::Device > devices;
	cl::Device device;
	int cl_device_type;
	cl_mem_flags cl_mem_extra_flags;
	bool cl_buffer_hostptr;
	// Context
	cl::Context context;
	// Buffer
	std::vector< cl::Buffer* > buffers;
	cl::Buffer *ret_buffer;
	size_t ret_buffer_size;
	void *ret_obj;
	// Program
	cl::Program program;
	const static std::string kernel_src;
	// Kernel
	cl::Kernel kernel;
	std::string kernel_name;
	size_t kernel_arg_idx;
	// CommandQueue
	cl::CommandQueue command_queue;


	Compute(std::string name, int device_type = CL_DEVICE_TYPE_CPU)
		: cl_device_type(device_type), kernel_name(name)
	{
		// Platform
		this->init_platform();

		// Device
		this->init_device();

		// Context
		this->init_context();

		// Program
		this->init_program();

		// Kernel
		this->init_kernel();

		// Command Queue
		this->init_command_queue();

		if (DEBUG)
		{
			this->clinfo();
		}
	}

	~Compute()
	{
		this->sync();
		this->reset_buffer();
	}

	void run(const int dm1, const int dm2 = 0, const int dm3 = 0)
	{
		cl::NDRange global_range;

		if (dm3 != 0)
			global_range = cl::NDRange(dm1, dm2, dm3);
		else if (dm2 != 0)
			global_range = cl::NDRange(dm1, dm2);
		else
			global_range = cl::NDRange(dm1);

		cl::Event event;
		this->command_queue.enqueueNDRangeKernel(
			this->kernel,
			cl::NullRange,  // offset
			global_range,  // global
			cl::NullRange,  // local
			NULL,  // events
			&event  // event
			);
		event.wait();

		// read result
		assert(this->ret_buffer != NULL);
		assert(this->ret_buffer_size != 0);
		this->command_queue.enqueueReadBuffer(
			*this->ret_buffer,
			true,  // block read
			0,  // offset
			this->ret_buffer_size,
			this->ret_obj,
			NULL,  // events
			&event
			);
	}

	void set_buffer(void *buffs,
		const size_t size,  // size in bytes
		cl_mem_flags flags = CL_MEM_READ_ONLY,
		cl::Buffer **ret = NULL)
	{
		cl_int err;

		flags |= this->cl_mem_extra_flags;

		cl::Buffer *buf = new cl::Buffer(
			this->context,
			flags,
			size,
			buffs,
			&err
			);

		this->buffers.push_back(buf);

		check_err(err, "cl::Buffer constructor");

		// set kernel argument
		if (DEBUG)
		{
			std::cout << "[DEBUG] cl::Kernel::setArg "
				<< this->kernel_arg_idx << ": "
				<< buf
				<< std::endl;
		}
		err = this->kernel.setArg(this->kernel_arg_idx, *buf);
		++(this->kernel_arg_idx);
		check_err(err, "cl::Kernel::Kernel");

		if (ret == NULL)
			return;

		*ret = buf;
	}

	template<typename T>
	void set_buffer(const T &val)
	{
		cl_int err;

		err = this->kernel.setArg(this->kernel_arg_idx, val);
		++(this->kernel_arg_idx);
		check_err(err, "cl::Kernel::Kernel");

		if (DEBUG)
		{
			std::cout << "[DEBUG] cl::Kernel::setArg "
				<< this->kernel_arg_idx << ": "
				<< "const setted"
				<< std::endl;
		}
	}

	void set_ret_buffer(void *buffs, const size_t size)
	{
		this->set_buffer(
			buffs,
			size,
			CL_MEM_WRITE_ONLY,
			&this->ret_buffer
			);
		this->ret_buffer_size = size;
		this->ret_obj = buffs;
	}

	void reset_buffer()
	{
		/* Release all cl buffer objs in vector `this->buffers`
		* Also, reset `ret_buffer` relatived vars.
		* */

		this->ret_buffer = NULL;
		this->ret_obj = NULL;
		this->ret_buffer_size = 0;

		for (auto &i : this->buffers)
			delete i;
		this->buffers.clear();
	}

	void reset_buffer(const size_t index, void *buffs)
	{
		/* Reuse the this->buffer[index]
		*
		* The size of `buffs` should be same as this->buffers[index].
		* */
		assert(index < this->buffers.size());
		assert(this->buffers[index] != NULL);

		size_t size;
		cl::Event event;
		cl_int error;

		this->buffers[index]->getInfo(CL_MEM_SIZE, &size);
		if (DEBUG)
		{
			std::cout << "[reset_buffer] Size of buffers"
				<< "[" << index << "]: "
				<< size << std::endl;
		}

		error = this->command_queue.enqueueWriteBuffer(
			*(this->buffers[index]),
			CL_TRUE,  // blocking
			0,  // offset
			size,
			buffs,
			NULL,  // events
			&event
			);
		event.wait();

		this->check_err(error, "CommandQueue::enqueueWriteBuffer");
	}

	void release_buffer(const size_t index)
	{
		assert(index < this->buffers.size());

		delete this->buffers[index];
		this->buffers[index] = NULL;
	}

	void sync()
	{
		cl_int error;

		error = this->command_queue.finish();
		this->check_err(error,
			"CommandQueue::finish CL_OUT_OF_HOST_MEMORY");
	}

	void clinfo()
	{
		size_t out;

		this->device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &out);
		std::cout << "Device Max Compute Units: "
			<< out << std::endl;
	}

private:

	void check_err(cl_int err, const char * name)
	{
		if (err == CL_SUCCESS)
			return;

		std::cerr << "[ERROR] " << name << " (" << err << ")" << std::endl;
		exit(1);
	}

	void init_platform()
	{
		cl::Platform::get(&this->platforms);
		this->check_err(platforms.size() ? CL_SUCCESS : -1, "cl::Platform::get");

		this->platform = platforms[0];

		// debug
		std::string name;
		platform.getInfo(CL_PLATFORM_NAME, &name);
		std::cout << "Platform:\t" << name << std::endl;
	}

	void init_device()
	{
		if (this->cl_device_type == CL_DEVICE_TYPE_CPU)
		{
			this->cl_mem_extra_flags = CL_MEM_USE_HOST_PTR;
			this->cl_buffer_hostptr = true;
		}
		else if (this->cl_device_type == CL_DEVICE_TYPE_GPU)
		{
			this->cl_mem_extra_flags = CL_MEM_COPY_HOST_PTR;
			this->cl_buffer_hostptr = false;
		}
		else
		{
			std::cout << "Wrong CL Device Type!" << std::endl;
			exit(1);
		}

		platform.getDevices(this->cl_device_type, &this->devices);
		this->check_err(devices.size() ? CL_SUCCESS : -1, "cl:Platform::getDevices");

		this->device = this->devices[0];

		// debug
		std::string dev_info;
		for (auto &dev : devices)
		{
			dev.getInfo(CL_DEVICE_NAME, &dev_info);
			std::cout << "Device:\t" << dev_info << std::endl;
		}
	}

	void init_context()
	{
		cl_int err;

		this->context = cl::Context::Context(devices,
			NULL,  // property
			NULL,  // call back
			NULL,  // user_data
			&err
			);
		check_err(err, "cl::Context::Context");
	}

	void init_program()
	{
		cl_int err;
		this->program = cl::Program::Program(
			context,
			this->kernel_src,
			true,  // build
			&err);

		// debug
		if (DEBUG)
		{
			std::string tmps;
			this->program.getBuildInfo(this->device, CL_PROGRAM_BUILD_LOG, &tmps);
			std::cout << "[cl::Program] Build log ==========================" << std::endl;
			std::cout << tmps << std::endl;
			std::cout << "[cl::Program] End of build log ===================" << std::endl;
			check_err(err, "cl::Program::Program");
		}
	}

	void save_binary_kernel(std::string path)
	{
		std::vector<char *> tmp_v;
		std::vector<size_t> tmp_size;
		std::ofstream f(path, std::ios::out | std::ios::binary);

		this->program.getInfo(CL_PROGRAM_BINARY_SIZES, &tmp_size);
		tmp_v.push_back(new char[tmp_size[0]]);

		this->program.getInfo(CL_PROGRAM_BINARIES, &tmp_v);
		std::cout << "[cl::Program] binaries: "
			<< tmp_v.size() << ", "
			<< "size: " << tmp_size[0]
			<< std::endl;

		f.write(tmp_v[0], tmp_size[0]);

		tmp_v.clear();
		f.close();
	}

	void init_kernel()
	{
		cl_int err;
		this->kernel = cl::Kernel::Kernel(program, this->kernel_name.c_str(), &err);
		this->kernel_arg_idx = 0;

		check_err(err, "cl::Kernel::Kernel");
	}

	void init_command_queue()
	{
		cl_int err;
		this->command_queue = cl::CommandQueue::CommandQueue(
			this->context,
			0,  // property
			&err);

		check_err(err, "cl::CommandQueue::CommandQueue");
	}
};

#endif /* end of include guard: COMPUTE_HPP */

#include "kernel.hpp"