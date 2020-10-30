//
// Created by b1xian on 2020-10-12.
//
#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

#include <time.h>

static void rgb_deinterleave_c(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* rgb, int len_color) {
    /*
     * Take the elements of "rgb" and store the individual colors "r", "g", and "b".
     */
    for (int i=0; i < len_color; i++) {
        r[i] = rgb[3*i];
        g[i] = rgb[3*i+1];
        b[i] = rgb[3*i+2];
    }
}

static void rgb_deinterleave_neon(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {
    /*
     * Take the elements of "rgb" and store the individual colors "r", "g", and "b"
     */
    int num8x16 = len_color / 16;
    uint8x16x3_t intlv_rgb;
    for (int i=0; i < num8x16; i++) {
        intlv_rgb = vld3q_u8(rgb+3*16*i);
        vst1q_u8(r+16*i, intlv_rgb.val[0]);
        vst1q_u8(g+16*i, intlv_rgb.val[1]);
        vst1q_u8(b+16*i, intlv_rgb.val[2]);
    }
}

int main(){
    int w = 1920;
    int h = 1080;
    int c = 3;

    int count = 1000;
    double none_neon_cost = 0.;
    double neon_cost = 0.;
    for (int k = 0; k < count; k++) {
        uint8_t *r;
        r = (uint8_t*)malloc(w*h);
        uint8_t *g;
        g = (uint8_t*)malloc(w*h);
        uint8_t *b;
        b = (uint8_t*)malloc(w*h);

        uint8_t* rgb;
        rgb = (uint8_t*)malloc(w*h*c);
//    for (int i = 0; i < h; i++) {
//        for (int j = 0; j < w; j++) {
//            rgb[(i*w+j)*3 + 0] = '1';
//            rgb[(i*w+j)*3 + 1] = '2';
//            rgb[(i*w+j)*3 + 2] = '3';
//        }
//    }
        for (int i=0; i < w*h; i++) {
            rgb[3*i]   = '1';
            rgb[3*i+1] = '2';
            rgb[3*i+2] = '3';
        }

        // change to chw
        clock_t start_time = clock();
        rgb_deinterleave_c(r, g, b, rgb, w*h);
        clock_t end_time = clock();
        none_neon_cost += (double)(end_time - start_time) / CLOCKS_PER_SEC;


        start_time = clock();
        rgb_deinterleave_neon(r, g, b, rgb, w*h);
        end_time = clock();
        neon_cost += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    }

    std::cout << "none_neon_cost: " << none_neon_cost / count << "ms" << std::endl;
    std::cout << "neon_cost: " << neon_cost / count << "ms" << std::endl;


    return 0;
}
