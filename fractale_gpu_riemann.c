#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

#define WIDTH 1220
#define HEIGHT 720
#define MAX_ITER 60
#define SUPERSAMPLING_FACTOR 1 // Facteur de suréchantillonnage


const char* kernelSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void fractal(__global unsigned char* image, __global int* accumulation, int width, int height,\n"
"                      double zoom, double centerX, double centerY, int maxIter,\n"
"                      int fractalType, int supersamplingFactor, double juliaRe, double juliaIm) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    int idx = (y * width + x) * 3;\n"
"    double aspect_ratio = (double)width / (double)height;\n"
"    double norm_x = ((double)x / width) * 2.0 - 1.0;\n"
"    double norm_y = ((double)y / height) * 2.0 - 1.0;\n"
"\n"
"    if (aspect_ratio > 1.0) {\n"
"        norm_x *= aspect_ratio;\n"
"    } else {\n"
"        norm_y /= aspect_ratio;\n"
"    }\n"
"\n"
"    double c_re = norm_x * zoom + centerX;\n"
"    double c_im = norm_y * zoom + centerY;\n"
"\n"
"    int iter = 0;\n"
"    double Z_re = c_re;\n"
"    double Z_im = c_im;\n"
"\n"
"    if (fractalType == 0) {\n"
"        while ((Z_re * Z_re + Z_im * Z_im <= 4.0) && (iter < maxIter)) {\n"
"            double temp = Z_re * Z_re - Z_im * Z_im + c_re;\n"
"            Z_im = 2.0 * Z_re * Z_im + c_im;\n"
"            Z_re = temp;\n"
"            iter++;\n"
"        }\n"
"    } else if (fractalType == 1) {\n"
"    } else if (fractalType == 2) {\n"
"        Z_re = norm_x * zoom + centerX;\n"
"        Z_im = norm_y * zoom + centerY;\n"
"        while ((Z_re * Z_re + Z_im * Z_im <= 4.0) && (iter < maxIter)) {\n"
"            double temp = Z_re * Z_re - Z_im * Z_im + juliaRe;\n"
"            Z_im = 2.0 * Z_re * Z_im + juliaIm;\n"
"            Z_re = temp;\n"
"            iter++;\n"
"        }\n"
"    } else if (fractalType == 3) {\n"
"        // Buddhabrot\n"
"        double initial_re = Z_re;\n"
"        double initial_im = Z_im;\n"
"\n"
"        // First pass: check for escape\n"
"        while ((Z_re * Z_re + Z_im * Z_im <= 4.0) && (iter < maxIter)) {\n"
"            double temp = Z_re * Z_re - Z_im * Z_im + initial_re;\n"
"            Z_im = 2.0 * Z_re * Z_im + initial_im;\n"
"            Z_re = temp;\n"
"            iter++;\n"
"        }\n"
"\n"
"        if (iter < maxIter) {\n"
"            Z_re = initial_re;\n"
"            Z_im = initial_im;\n"
"\n"
"            for (int i = 0; i < iter; i++) {\n"
"                double px_d = (Z_re - centerX) / zoom * width / 2.0 + width / 2.0;\n"
"                double py_d = (Z_im - centerY) / zoom * height / 2.0 + height / 2.0;\n"
"                int px = (int)(px_d + 0.5);\n"
"                int py = (int)(py_d + 0.5);\n"
"                if (px >= 0 && px < width && py >= 0 && py < height) {\n"
"                    int acc_idx = py * width + px;\n"
"                    atomic_add(&accumulation[acc_idx], 1);\n"
"                }\n"
"                double temp = Z_re * Z_re - Z_im * Z_im + initial_re;\n"
"                Z_im = 2.0 * Z_re * Z_im + initial_im;\n"
"                Z_re = temp;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    if (fractalType == 3) {\n"
"        int acc_idx = y * width + x;\n"
"        float norm_val = log((float)accumulation[acc_idx] + 1.0f) / log((float)maxIter + 1.0f);\n"
"        unsigned char grayscale = (unsigned char)(255 * norm_val);\n"
"        image[idx] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_val * 5.0));\n"
"        image[idx + 1] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_val * 5.0 + 2.0));\n"
"        image[idx + 2] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_val * 5.0 + 4.0));\n"
"    } else {\n"
"        double norm_iter = (double)iter / maxIter;\n"
"        image[idx] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_iter * 5.0));\n"
"        image[idx + 1] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_iter * 5.0 + 2.0));\n"
"        image[idx + 2] = (unsigned char)(128.0 + 127.0 * cos(3.0 + norm_iter * 5.0 + 4.0));\n"
"	     if (norm_iter == 1.0) {\n"	
"            image[idx] = 0;\n"
"            image[idx + 1] = 0;\n"
"            image[idx + 2] = 0;\n"
"        }\n"
"    }\n"
"}\n";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Erreur lors de %s (%d)\n", operation, err);
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Erreur d'initialisation de SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Fractale GPU", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Erreur de création de la fenêtre: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices, ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    checkError(ret, "clGetPlatformIDs");
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        checkError(ret, "clGetDeviceIDs");
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "clCreateContext");

    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    checkError(ret, "clCreateCommandQueueWithProperties");

    cl_mem image_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * 3, NULL, &ret);
    checkError(ret, "clCreateBuffer image_mem");

    cl_mem accumulation_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &ret);
    checkError(ret, "clCreateBuffer accumulation_mem");

    int* accumulation_init = (int*)calloc(WIDTH * HEIGHT, sizeof(int));
    ret = clEnqueueWriteBuffer(command_queue, accumulation_mem, CL_TRUE, 0, sizeof(int) * WIDTH * HEIGHT, accumulation_init, 0, NULL, NULL);
    checkError(ret, "clEnqueueWriteBuffer accumulation_mem");

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource");

    ret = clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("Erreur lors de la compilation du kernel :\n%s\n", build_log);
        free(build_log);
        return 1;
    }

    int buddhabrot_enabled = (argc > 1 && strcmp(argv[1], "buddhabrot") == 0);
    const char* kernel_name = buddhabrot_enabled ? "buddhabrot" : "mandelbrot";
    cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
    checkError(ret, "clCreateKernel");

    double zoom = 4.0 / WIDTH;
    double offsetX = -0.7;
    double offsetY = 0.0;
    int width = WIDTH;
    int height = HEIGHT;
    int maxIter = MAX_ITER;

    int running = 1;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = 0;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_SPACE) {
                    maxIter += 1000;
                    printf("maxIter augmenté à %d\n", maxIter);
                }
            }
            if (event.type == SDL_MOUSEWHEEL) {
                double mouseX, mouseY;
                int x, y;
                SDL_GetMouseState(&x, &y);
                mouseX = (double)x;
                mouseY = (double)y;
                double mouseRe = (mouseX - width / 2.0) * zoom + offsetX;
                double mouseIm = (mouseY - height / 2.0) * zoom + offsetY;
                if (event.wheel.y > 0) zoom /= 1.1;
                else if (event.wheel.y < 0) zoom *= 1.1;
                offsetX = mouseRe - (mouseX - width / 2.0) * zoom;
                offsetY = mouseIm - (mouseY - height / 2.0) * zoom;
            }
            if (event.type == SDL_MOUSEMOTION) {
                if (event.motion.state & SDL_BUTTON_LMASK) {
                    offsetX -= event.motion.xrel * zoom;
                    offsetY -= event.motion.yrel * zoom;
                }
            }
        }

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_mem);
        checkError(ret, "clSetKernelArg 0");
        if (buddhabrot_enabled) {
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &accumulation_mem);
            checkError(ret, "clSetKernelArg 1 accumulation_mem");
        }
        ret = clSetKernelArg(kernel, 1 + buddhabrot_enabled, sizeof(int), &width);
        checkError(ret, "clSetKernelArg 1/2 width");
        ret = clSetKernelArg(kernel, 2 + buddhabrot_enabled, sizeof(int), &height);
        checkError(ret, "clSetKernelArg 2/3 height");
        ret = clSetKernelArg(kernel, 3 + buddhabrot_enabled, sizeof(double), &zoom);
        checkError(ret, "clSetKernelArg 3/4 zoom");
        ret = clSetKernelArg(kernel, 4 + buddhabrot_enabled, sizeof(double), &offsetX);
        checkError(ret, "clSetKernelArg 4/5 offsetX");
        ret = clSetKernelArg(kernel, 5 + buddhabrot_enabled, sizeof(double), &offsetY);
        checkError(ret, "clSetKernelArg 5/6 offsetY");
        ret = clSetKernelArg(kernel, 6 + buddhabrot_enabled, sizeof(int), &maxIter);
        checkError(ret, "clSetKernelArg 6/7 maxIter");

        size_t global_work_size[] = { (size_t)WIDTH, (size_t)HEIGHT };
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        checkError(ret, "clEnqueueNDRangeKernel");

        ret = clEnqueueReadBuffer(command_queue, image_mem, CL_TRUE, 0, WIDTH * HEIGHT * 3, image, 0, NULL, NULL);
        checkError(ret, "clEnqueueReadBuffer");

        SDL_UpdateTexture(texture, NULL, image, WIDTH * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(image_mem);
    clReleaseMemObject(accumulation_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(image);
    free(accumulation_init);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
