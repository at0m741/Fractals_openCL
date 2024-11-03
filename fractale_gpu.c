#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITER 1000

const char* kernelSource =

"__constant double palette[16][3] = {\n"
"    {0.2588f, 0.1176f, 0.0588f},\n"
"    {0.0980f, 0.0275f, 0.1020f},\n"
"    {0.0353f, 0.0039f, 0.1843f},\n"
"    {0.0157f, 0.0157f, 0.2863f},\n"
"    {0.0f, 0.0275f, 0.3922f},\n"
"    {0.0471f, 0.1725f, 0.5412f},\n"
"    {0.0941f, 0.3216f, 0.6941f},\n"
"    {0.2235f, 0.4902f, 0.8196f},\n"
"    {0.5255f, 0.7098f, 0.8980f},\n"
"    {0.8275f, 0.9255f, 0.9725f},\n"
"    {0.9451f, 0.9137f, 0.7490f},\n"
"    {0.9725f, 0.7882f, 0.3725f},\n"
"    {1.0f, 0.6667f, 0.0f},\n"
"    {0.8000f, 0.5020f, 0.0f},\n"
"    {0.6000f, 0.3412f, 0.0f},\n"
"    {0.4157f, 0.2039f, 0.0118f}\n"
"};\n"

"__kernel void mandelbrot(__global unsigned char* image, int width, int height,\n"
"                         double zoom, double offsetX, double offsetY, int maxIter) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"    double c_re = (x - width / 2.0) * zoom + offsetX;\n"
"    double c_im = (y - height / 2.0) * zoom + offsetY;\n"
"    double Z_re = c_re, Z_im = c_im;\n"
"    int iter = 0;\n"
"    double Z_re2 = Z_re * Z_re;\n"
"    double Z_im2 = Z_im * Z_im;\n"
"    while ((Z_re2 + Z_im2 <= 4.0) && (iter < maxIter)) {\n"
"        Z_im = 2.0 * Z_re * Z_im + c_im;\n"
"        Z_re = Z_re2 - Z_im2 + c_re;\n"
"        Z_re2 = Z_re * Z_re;\n"
"        Z_im2 = Z_im * Z_im;\n"
"        iter++;\n"
"    }\n"
"    int idx = (y * width + x) * 3;\n"
"    if (iter == maxIter) {\n"
"        image[idx] = 0;\n"
"        image[idx + 1] = 0;\n"
"        image[idx + 2] = 0;\n"
"    } else {\n"
"        double log_zn = log(Z_re2 + Z_im2) / 2.0;\n"
"        double nu = log(log_zn / log(2.0)) / log(2.0);\n"
"        double smooth_iter = iter + 1 - nu;\n"
"        double t = smooth_iter / 250 * 16.0;\n"
"        int colorIdx1 = (int)floor(t) % 16;\n"
"        int colorIdx2 = (colorIdx1 + 1) % 16;\n"
"        double frac = t - floor(t);\n"
"        double r = palette[colorIdx1][0] * (1 - frac) + palette[colorIdx2][0] * frac;\n"
"        double g = palette[colorIdx1][1] * (1 - frac) + palette[colorIdx2][1] * frac;\n"
"        double b = palette[colorIdx1][2] * (1 - frac) + palette[colorIdx2][2] * frac;\n"
"        image[idx] = (unsigned char)(r * 255);\n"
"        image[idx + 1] = (unsigned char)(g * 255);\n"
"        image[idx + 2] = (unsigned char)(b * 255);\n"
"    }\n"
"}\n"

"__kernel void buddhabrot(__global unsigned char* image, __global int* accumulation, int width, int height,\n"
"                         double zoom, double offsetX, double offsetY, int maxIter) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"    double c_re = (x - width / 2.0) * zoom + offsetX;\n"
"    double c_im = (y - height / 2.0) * zoom + offsetY;\n"
"    double Z_re = c_re, Z_im = c_im;\n"
"    int iter = 0;\n"
"    double Z_re2 = Z_re * Z_re;\n"
"    double Z_im2 = Z_im * Z_im;\n"
"    while ((Z_re2 + Z_im2 <= 4.0) && (iter < maxIter)) {\n"
"        Z_im = 2.0 * Z_re * Z_im + c_im;\n"
"        Z_re = Z_re2 - Z_im2 + c_re;\n"
"        Z_re2 = Z_re * Z_re;\n"
"        Z_im2 = Z_im * Z_im;\n"
"        iter++;\n"
"    }\n"
"    if (iter < maxIter) {\n"
"        Z_re = c_re;\n"
"        Z_im = c_im;\n"
"        for (int i = 0; i < iter; i++) {\n"
"            Z_im = 2.0 * Z_re * Z_im + c_im;\n"
"            Z_re = Z_re2 - Z_im2 + c_re;\n"
"            Z_re2 = Z_re * Z_re;\n"
"            Z_im2 = Z_im * Z_im;\n"
"            int px = (int)((Z_re - offsetX) / zoom + width / 2.0);\n"
"            int py = (int)((Z_im - offsetY) / zoom + height / 2.0);\n"
"            if (px >= 0 && px < width && py >= 0 && py < height) {\n"
"                atomic_add(&accumulation[py * width + px], 1);\n"
"            }\n"
"        }\n"
"    }\n"
"    int idx = (y * width + x) * 3;\n"
"    int acc_idx = y * width + x;\n"
"    int acc_val = accumulation[acc_idx];\n"
"    double norm_val = log((double)acc_val + 1.0) / log((double)maxIter + 1.0);\n"
"    double t = norm_val * 16.0;\n"
"    int colorIdx1 = (int)floor(t) % 16;\n"
"    int colorIdx2 = (colorIdx1 + 1) % 16;\n"
"    double frac = t - floor(t);\n"
"    double r = palette[colorIdx1][0] * (1 - frac) + palette[colorIdx2][0] * frac;\n"
"    double g = palette[colorIdx1][1] * (1 - frac) + palette[colorIdx2][1] * frac;\n"
"    double b = palette[colorIdx1][2] * (1 - frac) + palette[colorIdx2][2] * frac;\n"
"    image[idx] = (unsigned char)(r * 255);\n"
"    image[idx + 1] = (unsigned char)(g * 255);\n"
"    image[idx + 2] = (unsigned char)(b * 255);\n"
"}\n"

"__kernel void mandelbulb(__global unsigned char* image, int width, int height,\n"
"                         double zoom, double offsetX, double offsetY, double offsetZ, int maxIter) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"    double aspect_ratio = (double)width / (double)height;\n"
"    double norm_x = ((double)x / width) * 2.0 - 1.0;\n"
"    double norm_y = ((double)y / height) * 2.0 - 1.0;\n"
"    if (aspect_ratio > 1.0) {\n"
"        norm_x *= aspect_ratio;\n"
"    } else {\n"
"        norm_y /= aspect_ratio;\n"
"    }\n"
"    double c_re = norm_x * zoom + offsetX;\n"
"    double c_im = norm_y * zoom + offsetY;\n"
"    double c_z = offsetZ;\n"
"    double Z_re = c_re;\n"
"    double Z_im = c_im;\n"
"    double Z_z = c_z;\n"
"    int iter = 0;\n"
"    while ((Z_re * Z_re + Z_im * Z_im + Z_z * Z_z <= 4.0) && (iter < maxIter)) {\n"
"        double r = sqrt(Z_re * Z_re + Z_im * Z_im + Z_z * Z_z);\n"
"        double theta = atan2(sqrt(Z_re * Z_re + Z_im * Z_im), Z_z);\n"
"        double phi = atan2(Z_im, Z_re);\n"
"        double new_r = pow(r, 8.0);\n"
"        double new_theta = theta * 8.0;\n"
"        double new_phi = phi * 8.0;\n"
"        Z_re = new_r * sin(new_theta) * cos(new_phi) + c_re;\n"
"        Z_im = new_r * sin(new_theta) * sin(new_phi) + c_im;\n"
"        Z_z = new_r * cos(new_theta) + c_z;\n"
"        iter++;\n"
"    }\n"
"    int idx = (y * width + x) * 3;\n"
"    if (iter == maxIter) {\n"
"        image[idx] = 0;\n"
"        image[idx + 1] = 0;\n"
"        image[idx + 2] = 0;\n"
"    } else {\n"
"        double norm_iter = (double)iter / maxIter;\n"
"        double t = norm_iter * 16.0;\n"
"        int colorIdx1 = (int)floor(t) % 16;\n"
"        int colorIdx2 = (colorIdx1 + 1) % 16;\n"
"        double frac = t - floor(t);\n"
"        double r = palette[colorIdx1][0] * (1 - frac) + palette[colorIdx2][0] * frac;\n"
"        double g = palette[colorIdx1][1] * (1 - frac) + palette[colorIdx2][1] * frac;\n"
"        double b = palette[colorIdx1][2] * (1 - frac) + palette[colorIdx2][2] * frac;\n"
"        image[idx] = (unsigned char)(r * 255);\n"
"        image[idx + 1] = (unsigned char)(g * 255);\n"
"        image[idx + 2] = (unsigned char)(b * 255);\n"
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

    cl_mem accumulation_mem = NULL;
    if (argc > 1 && strcmp(argv[1], "buddhabrot") == 0) {
        accumulation_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &ret);
        checkError(ret, "clCreateBuffer accumulation_mem");
        int* accumulation_init = (int*)calloc(WIDTH * HEIGHT, sizeof(int));
        ret = clEnqueueWriteBuffer(command_queue, accumulation_mem, CL_TRUE, 0, sizeof(int) * WIDTH * HEIGHT, accumulation_init, 0, NULL, NULL);
        free(accumulation_init);
        checkError(ret, "clEnqueueWriteBuffer accumulation_mem");
    }

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

    int fractal_type = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "mandelbrot") == 0) {
            fractal_type = 0;
        } else if (strcmp(argv[1], "buddhabrot") == 0) {
            fractal_type = 1;
        } else if (strcmp(argv[1], "mandelbulb") == 0) {
            fractal_type = 2;
        } else {
            printf("Usage: %s [mandelbrot|buddhabrot|mandelbulb]\n", argv[0]);
            return 1;
        }
    } else {
        printf("Usage: %s [mandelbrot|buddhabrot|mandelbulb]\n", argv[0]);
        return 1;
    }

    const char* kernel_name = (fractal_type == 0) ? "mandelbrot" :
                              (fractal_type == 1) ? "buddhabrot" : "mandelbulb";
    cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
    checkError(ret, "clCreateKernel");

    double zoom = 4.0 / WIDTH;
    double offsetX = -0.7, offsetY = 1.0, offsetZ = 0.0;
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
                if (fractal_type == 2) {
                    if (event.key.keysym.sym == SDLK_UP) {
                        offsetZ += 0.1;
                    } else if (event.key.keysym.sym == SDLK_DOWN) {
                        offsetZ -= 0.1;
                    }
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

        if (fractal_type == 1) {
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &accumulation_mem);
            checkError(ret, "clSetKernelArg 1 accumulation_mem");
        }

        ret = clSetKernelArg(kernel, 1 + (fractal_type == 1), sizeof(int), &width);
        checkError(ret, "clSetKernelArg 1/2 width");
        ret = clSetKernelArg(kernel, 2 + (fractal_type == 1), sizeof(int), &height);
        checkError(ret, "clSetKernelArg 2/3 height");
        ret = clSetKernelArg(kernel, 3 + (fractal_type == 1), sizeof(double), &zoom);
        checkError(ret, "clSetKernelArg 3/4 zoom");
        ret = clSetKernelArg(kernel, 4 + (fractal_type == 1), sizeof(double), &offsetX);
        checkError(ret, "clSetKernelArg 4/5 offsetX");
        ret = clSetKernelArg(kernel, 5 + (fractal_type == 1), sizeof(double), &offsetY);
        checkError(ret, "clSetKernelArg 5/6 offsetY");



if (fractal_type == 2) { // Mandelbulb
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

    ret = clSetKernelArg(kernel, 6, sizeof(double), &offsetZ);
    checkError(ret, "clSetKernelArg 6");

    ret = clSetKernelArg(kernel, 7, sizeof(int), &maxIter);
    checkError(ret, "clSetKernelArg 7");
}



        ret = clSetKernelArg(kernel, 6 + (fractal_type == 1), sizeof(int), &maxIter);
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
    if (accumulation_mem) clReleaseMemObject(accumulation_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(image);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
