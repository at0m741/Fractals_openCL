#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITER 1000

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Erreur lors de %s (%d)\n", operation, err);
        exit(1);
    }
}

char* readKernelSource(const char* filename) {
    FILE* fp;
    size_t source_size;
    char* source_str;

    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier kernel.\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);

    source_str = (char*)malloc(source_size + 1);
    source_str[source_size] = '\0';
    fread(source_str, sizeof(char), source_size, fp);
    fclose(fp);

    return source_str;
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

    char* kernelSource = readKernelSource("kernel.cl");

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &ret);
    checkError(ret, "clCreateProgramWithSource");

    ret = clBuildProgram(program, 1, &device_id, 
    "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros",
	NULL, NULL);
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

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &ret);
    checkError(ret, "clCreateKernel");

    double zoom = 4.0f / WIDTH;
    double offsetX = -0.7, offsetY = 0.0;
	double centerX = -0.7f; 
	double centerY = 0.0f;
    int width = WIDTH;
    int height = HEIGHT;
    int maxIter = MAX_ITER;
    int samples = 2;

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

        ret = clSetKernelArg(kernel, 1, sizeof(int), &width);
        checkError(ret, "clSetKernelArg 1");

        ret = clSetKernelArg(kernel, 2, sizeof(int), &height);
        checkError(ret, "clSetKernelArg 2");

        ret = clSetKernelArg(kernel, 3, sizeof(double), &zoom);
        checkError(ret, "clSetKernelArg 3");

        ret = clSetKernelArg(kernel, 4, sizeof(double), &offsetX);
        checkError(ret, "clSetKernelArg 4");

        ret = clSetKernelArg(kernel, 5, sizeof(double), &offsetY);
        checkError(ret, "clSetKernelArg 5");

        ret = clSetKernelArg(kernel, 6, sizeof(int), &maxIter);
        checkError(ret, "clSetKernelArg 6");

        ret = clSetKernelArg(kernel, 7, sizeof(int), &samples);
        checkError(ret, "clSetKernelArg 7");

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
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(kernelSource);
    free(image);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
