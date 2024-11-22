
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define FIXED_BITS 24
#define FIXED_ONE (1 << FIXED_BITS)

int to_fixed(float value) {
    return (int)(value * FIXED_ONE);
}

float to_float(int value) {
    return (float)value / FIXED_ONE;
}

int fixed_mul(int a, int b) {
    return (int)(((long long)a * b) >> FIXED_BITS);
}

int fixed_div(int a, int b) {
    return (int)(((long long)a << FIXED_BITS) / b);
}
__constant float palette[16] = {
    (float)(0.2588, 0.1176, 0.0588),
    (float)(0.0980, 0.0275, 0.1020),
    (float)(0.0353, 0.0039, 0.1843),
    (float)(0.0157, 0.0157, 0.2863),
    (float)(0.0, 0.0275, 0.3922),
    (float)(0.0471, 0.1725, 0.5412),
    (float)(0.0941, 0.3216, 0.6941),
    (float)(0.2235, 0.4902, 0.8196),
    (float)(0.5255, 0.7098, 0.8980),
    (float)(0.8275, 0.9255, 0.9725),
    (float)(0.9451, 0.9137, 0.7490),
    (float)(0.9725, 0.7882, 0.3725),
    (float)(1.0, 0.6667, 0.0),
    (float)(0.8000, 0.5020, 0.0),
    (float)(0.6000, 0.3412, 0.0),
    (float)(0.4157, 0.2039, 0.0118)
};

__kernel void mandelbrot(__global unsigned char* image, int width, int height,
                         float zoom, float offsetX, float offsetY, int maxIter, int samples) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float color_r = 0.0f, color_g = 0.0f, color_b = 0.0f;

    double half_width = width * 0.5;
    double half_height = height * 0.5;
    double base_c_re = ((double)x - half_width) * zoom + offsetX;
    double base_c_im = ((double)y - half_height) * zoom + offsetY;
    double delta = (double)zoom / samples;

    for (int i = 0; i < samples; i++) {
        double c_re = base_c_re + ((double)i + 0.5) * delta;
        for (int j = 0; j < samples; j++) {
            double c_im = base_c_im + ((double)j + 0.5) * delta;

            double Z_re = c_re, Z_im = c_im;
            int iter = 0;
            while ((Z_re * Z_re + Z_im * Z_im <= 4.0) && (iter < maxIter)) {
                double Z_re_temp = Z_re * Z_re - Z_im * Z_im + c_re;
                Z_im = 2.0 * Z_re * Z_im + c_im;
                Z_re = Z_re_temp;
                iter++;
            }

            if (iter < maxIter) {
                float smooth_iter = iter + 1 - log(log(Z_re * Z_re + Z_im * Z_im) / log(2.0)) / log(2.0);
                float t = smooth_iter / 250.0f * 16.0f;
                int colorIdx1 = (int)t % 16;
                int colorIdx2 = (colorIdx1 + 1) % 16;
                float frac = t - floor(t);

                float3 color1 = palette[colorIdx1];
                float3 color2 = palette[colorIdx2];
                float3 color = color1 * (1.0f - frac) + color2 * frac;

                color_r += color.x;
                color_g += color.y;
                color_b += color.z;
            }
        }
    }

    color_r /= (samples * samples);
    color_g /= (samples * samples);
    color_b /= (samples * samples);

    int idx = (y * width + x) * 3;
    image[idx] = (unsigned char)(fmin(color_r * 255.0f, 255.0f));
    image[idx + 1] = (unsigned char)(fmin(color_g * 255.0f, 255.0f));
    image[idx + 2] = (unsigned char)(fmin(color_b * 255.0f, 255.0f));
}
