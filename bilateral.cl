
double gDistance(int x, int y, int sigma_s) {
    return exp((double) (-(x * x + y * y) / (2 * sigma_s * sigma_s)));
}

double gIntensity(int v, int sigma_v) {
    return exp((double) (-(v * v) / (2 * sigma_v * sigma_v)));
}

int gClamp(int x, int minval, int maxval) {
    return min(max(x, minval), maxval);
}

__kernel void bilateral(__global unsigned char *image_in, __global unsigned char *image_out,
                        const int width, const int height, const int w, const int sigma_v, const int sigma_s) {
    size_t g_id = get_global_id(0);
    if ((g_id >= width * height) || (g_id < 0)) {
        return;
    }
    const int x = g_id % width;
    const int y = g_id / width;

    image_out[g_id * 4 + 3] = image_in[g_id * 4 + 3]; // A
    // image_out[g_id * 4 + 2] = image_in[g_id * 4 + 2]; // R
    // image_out[g_id * 4 + 1] = image_in[g_id * 4 + 1]; // G
    // image_out[g_id * 4 + 0] = image_in[g_id * 4 + 0]; // B

    double FR = 0;
    double FG = 0;
    double FB = 0;
    double WR = 0;
    double WG = 0;
    double WB = 0;

    __global unsigned char* currentPixel = &image_in[(gClamp(y, 0, height - 1) * width + gClamp(x, 0, width - 1)) * 4];

    const int pixelStartX = x - w;
    const int pixelEndX = x + w;
    const int pixelStartY = y - w;
    const int pixelEndY = y + w;
    // If I insert ^ these directly into the loop, it glitches out the edges?
    for (int r = pixelStartX; r < pixelEndX; r++) {
        for (int s = pixelStartY; s < pixelEndY; s++) {
            __global unsigned char* pixelNeighbor = &image_in[(gClamp(s, 0, height - 1) * width + gClamp(r, 0, width - 1)) * 4];
            double gs = gDistance(abs_diff(r, x), abs_diff(s, y), sigma_s);
            double tR = gs * gIntensity(abs_diff(*(currentPixel + 2), *(pixelNeighbor + 2)), sigma_v);
            double tG = gs * gIntensity(abs_diff(*(currentPixel + 1), *(pixelNeighbor + 1)), sigma_v);
            double tB = gs * gIntensity(abs_diff(*(currentPixel + 0), *(pixelNeighbor + 0)), sigma_v);
            FR += *(pixelNeighbor + 2) * tR;
            FG += *(pixelNeighbor + 1) * tG;
            FB += *(pixelNeighbor + 0) * tB;
            WR += tR;
            WG += tG;
            WB += tB;
        }
    }

    image_out[g_id * 4 + 2] = FR / WR; // R
    image_out[g_id * 4 + 1] = FG / WG; // G
    image_out[g_id * 4 + 0] = FB / WB; // B
}
