/* embed_encode.c — encode the raw fp16 token-embedding table (262144 x 5376)
 * into 36-byte median-law binary glyphs (same law as encode_glyph_36),
 * so the embedding joins the RAM-fit stream instead of 2.82 GB fp16.
 * usage: embed_encode <embed_tokens.bin (fp16)> <d_out> <d_in> <out.bin>
 * verify mode: embed_encode verify <embed.bin> <glyphs.bin> <d_out> <d_in> [rows]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static inline uint16_t f32_to_f16(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000,e=(x>>23)&0xFF,m=x&0x7FFFFF;
    if(e==0) return (uint16_t)s;
    if(e==255) return (uint16_t)(s|0x7C00|(m?0x200:0));
    int e2=(int)e-127+15;
    if(e2>=31) return (uint16_t)(s|0x7C00);
    if(e2<=0) return (uint16_t)s;
    return (uint16_t)(s|(uint32_t)e2<<10|(m>>13));
}
static inline float f16_to_f32(uint16_t h){
    uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF;
    uint32_t x;
    if(e==0) x=s;
    else if(e==31) x=s|0x7F800000|(m<<13);
    else x=s|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&x,4); return f;
}

static void enc_glyph_36(const float *w, uint8_t *out){
    float s[256]; memcpy(s,w,256*sizeof(float));
    for(int i=1;i<256;i++){ float v=s[i]; int j=i-1; while(j>=0&&s[j]>v){s[j+1]=s[j];j--;} s[j+1]=v; }
    float median=(s[127]+s[128])*0.5f;
    float amax=0.f;
    for(int i=0;i<256;i++){ float c=fabsf(w[i]-median); if(c>amax)amax=c; }
    float scale=amax>0?amax:1.f;
    uint16_t hs=f32_to_f16(scale), hb=f32_to_f16(median);
    memcpy(out,&hs,2); memcpy(out+2,&hb,2);
    memset(out+4,0,32);
    for(int i=0;i<256;i++) if(w[i]-median>=0.f) out[4+(i>>3)]|=(uint8_t)(0x80>>(i&7));
}
static void dec_glyph_36(const uint8_t *gl, float *out){
    float scale=f16_to_f32(*(const uint16_t*)(gl+0));
    float bias=f16_to_f32(*(const uint16_t*)(gl+2));
    for(int i=0;i<256;i++){
        int bit=(gl[4+(i>>3)]>>(7-(i&7)))&1;
        out[i]=bias+(bit?scale:-scale);
    }
}

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: embed_encode encode|verify ...\n"); return 1; }
    if(!strcmp(argv[1],"encode")){
        if(argc<6){ fprintf(stderr,"usage: embed_encode encode <emb.bin> <d_out> <d_in> <out.bin>\n"); return 1; }
        FILE *f=fopen(argv[2],"rb"); if(!f){fprintf(stderr,"FATAL %s\n",argv[2]);return 1;}
        int d_out=atoi(argv[3]), d_in=atoi(argv[4]);
        int gpr=(d_in+255)/256;
        FILE *o=fopen(argv[5],"wb"); if(!o){fprintf(stderr,"FATAL %s\n",argv[5]);return 1;}
        uint16_t *row=malloc((size_t)d_in*2);
        float *w=malloc(256*sizeof(float));
        uint8_t *gl=malloc(36);
        long total=0;
        for(int r=0;r<d_out;r++){
            if(fread(row,2,d_in,f)!=(size_t)d_in){ fprintf(stderr,"FATAL: short read row %d\n",r); return 1; }
            for(int g=0;g<gpr;g++){
                int col=g*256, rem=d_in-col, n=rem>256?256:rem; if(n<0)n=0;
                for(int i=0;i<n;i++) w[i]=f16_to_f32(row[col+i]);
                for(int i=n;i<256;i++) w[i]=0.f;
                enc_glyph_36(w,gl);
                fwrite(gl,1,36,o);
                total++;
            }
            if(r%32768==0){ printf("row %d/%d (%.1f%%)\n",r,d_out,100.0*r/d_out); fflush(stdout); }
        }
        fclose(f); fclose(o);
        printf("DONE: %ld glyphs, %.1f MB -> %s\n", total, total*36/1e6, argv[5]);
        return 0;
    }
    if(!strcmp(argv[1],"verify")){
        if(argc<7){ fprintf(stderr,"usage: embed_encode verify <emb.bin> <glyphs.bin> <d_out> <d_in> [rows]\n"); return 1; }
        int d_out=atoi(argv[4]), d_in=atoi(argv[5]);
        int rows=argc>6?atoi(argv[6]):256;
        if(rows>d_out) rows=d_out;
        int gpr=(d_in+255)/256;
        FILE *f=fopen(argv[2],"rb"), *g=fopen(argv[3],"rb");
        fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET);
        if(esz!=(long)d_out*d_in*2){ fprintf(stderr,"FATAL: embed size %ld != %ld\n",esz,(long)d_out*d_in*2); return 1; }
        uint16_t *row=malloc((size_t)d_in*2);
        float w[256], wd[256];
        double sse=0, sst=0;
        for(int r=0;r<rows;r++){
            fread(row,2,d_in,f);
            for(int gg=0;gg<gpr;gg++){
                int col=gg*256, rem=d_in-col, n=rem>256?256:rem; if(n<0)n=0;
                for(int i=0;i<n;i++) w[i]=f16_to_f32(row[col+i]);
                for(int i=n;i<256;i++) w[i]=0.f;
                uint8_t gl[36]; fread(gl,1,36,g);
                dec_glyph_36(gl,wd);
                for(int i=0;i<n;i++){ double d=w[i]-wd[i]; sse+=d*d; sst+=w[i]*w[i]; }
            }
        }
        printf("verify %d rows: SNR = %.2f dB (36B binary vs fp16 embed)\n", rows, 10*log10(sst/(sse+1e-30)));
        return 0;
    }
    return 1;
}
