/* reencode31b.c — RAM-fit re-encoder for the Gemma-4-31B Language-U sandwich stream.
 *
 * Reads the official 7.08 GB stream (OPT4 132B boundary + OPT5 48B middle glyphs),
 * re-quantizes every glyph to either:
 *   - 92 B packed 11-bit dual-stage RVQ (lossless vs the 132B doctrine container,
 *     indices packed 16->11 bits: 4 + 32*2*11/8 = 92 B), or
 *   - 36 B median-law binary glyph (encode_glyph_36 law: w = median +- scale,
 *     256 sign bits MSB-first big-endian),
 * choosing per projection to maximize fidelity under a byte budget (default 4.76 GB).
 *
 * Laws implemented exactly as in Language-U-Gemma4/compiler/glyphs.py and
 * Language-U-Transformer/langut/runtime/cortex.py:
 *   132B: [s1 f16][s2 f16][32 u16 idx1][32 u16 idx2]; per octonion: fwht8(s1*cb1+s2*cb2)
 *   48B : [scale f16][bias f16][32B bitmask][4 u8 outlier_idx][4 f16 outlier_res]
 *         w[i] = bias + (bit? +scale : -scale) + sum_k res_k * (i==idx_k)
 *   36B : [scale f16][median f16][32B bitmask]; w[i] = median + (bit? +scale : -scale)
 *
 * usage: reencode31b encode <stream.bin> <layout.json> <codebook.bin> <out_base> [budget_bytes]
 *        reencode31b verify <orig_stream> <orig_layout> <new_stream> <new_layout> [max_proj]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

/* ---- standalone fp16 helpers (no repo dependency) ---- */
static inline uint16_t fp32_to_fp16(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000,e=(x>>23)&0xFF,m=x&0x7FFFFF;
    if(e==0) return (uint16_t)s;
    if(e==255) return (uint16_t)(s|0x7C00|(m?0x200:0));
    int e2=(int)e-127+15;
    if(e2>=31) return (uint16_t)(s|0x7C00);
    if(e2<=0) return (uint16_t)s;
    return (uint16_t)(s|(uint32_t)e2<<10|(m>>13));
}
static inline float fp16_to_fp32(uint16_t h){
    uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF;
    uint32_t x;
    if(e==0) x=s; else if(e==31) x=s|0x7F800000|(m<<13); else x=s|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&x,4); return f;
}
typedef struct { uint32_t magic; uint16_t version; uint16_t codec; uint32_t glyph_count; uint16_t d_model; uint16_t layers; } LanuHeader;
#define LANU_MAGIC 0x554e414cu

/* ---------- fwht8 (normalized Hadamard = H8, symmetric) ---------- */
static void fwht8(const float *in, float *out){
    const float inv = 0.3535533905932737622f;
    float a0=in[0]+in[1], a1=in[0]-in[1], a2=in[2]+in[3], a3=in[2]-in[3];
    float a4=in[4]+in[5], a5=in[4]-in[5], a6=in[6]+in[7], a7=in[6]-in[7];
    float b0=a0+a2, b2=a0-a2, b1=a1+a3, b3=a1-a3;
    float b4=a4+a6, b6=a4-a6, b5=a5+a7, b7=a5-a7;
    out[0]=(b0+b4)*inv; out[4]=(b0-b4)*inv;
    out[1]=(b1+b5)*inv; out[5]=(b1-b5)*inv;
    out[2]=(b2+b6)*inv; out[6]=(b2-b6)*inv;
    out[3]=(b3+b7)*inv; out[7]=(b3-b7)*inv;
}

/* ---------- glyph decoders (orig stream) ---------- */
/* 132B doctrine RVQ glyph -> 256 weights */
static void dec_glyph_132(const uint8_t *gl, const float *cb, float *out){
    float s1=fp16_to_fp32(*(const uint16_t*)(gl+0));
    float s2=fp16_to_fp32(*(const uint16_t*)(gl+2));
    for(int oct=0;oct<32;oct++){
        uint16_t i1,i2; memcpy(&i1,gl+4+oct*2,2); memcpy(&i2,gl+68+oct*2,2);
        const float *c1=cb+(size_t)(i1&2047)*8, *c2=cb+(size_t)(i2&2047)*8;
        float temp[8];
        for(int t=0;t<8;t++) temp[t]=s1*c1[t]+s2*c2[t];
        fwht8(temp, out+oct*8);
    }
}
/* 48B binary+outliers glyph -> 256 weights */
static void dec_glyph_48(const uint8_t *gl, const float *x, float *out_w /*unused*/, float *dot_out){
    /* we only need the decoded weights for requant; compute w[256] */
    float scale=fp16_to_fp32(*(const uint16_t*)(gl+0));
    float bias=fp16_to_fp32(*(const uint16_t*)(gl+2));
    (void)x;
    for(int i=0;i<256;i++){
        int bit=(gl[4+(i>>3)]>>(7-(i&7)))&1;
        out_w[i]=bias+(bit?scale:-scale);
    }
    for(int k=0;k<4;k++){
        uint8_t oi=gl[36+k];
        float res=fp16_to_fp32(*(const uint16_t*)(gl+40+k*2));
        if(oi<256) out_w[oi]+=res;
    }
    (void)dot_out;
}
/* 36B median-law glyph -> 256 weights */
static void dec_glyph_36(const uint8_t *gl, float *out){
    float scale=fp16_to_fp32(*(const uint16_t*)(gl+0));
    float bias=fp16_to_fp32(*(const uint16_t*)(gl+2));
    for(int i=0;i<256;i++){
        int bit=(gl[4+(i>>3)]>>(7-(i&7)))&1;
        out[i]=bias+(bit?scale:-scale);
    }
}
/* 92B packed 11-bit RVQ glyph -> 256 weights */
static void dec_glyph_92(const uint8_t *gl, const float *cb, float *out){
    float s1=fp16_to_fp32(*(const uint16_t*)(gl+0));
    float s2=fp16_to_fp32(*(const uint16_t*)(gl+2));
    /* 64 x 11-bit indices packed MSB-first into gl+4..gl+92 (704 bits = 88 B) */
    uint32_t idx[64];
    { long bitpos=0;
      for(int k=0;k<64;k++){
          long byte=4+bitpos/8; int off=(int)(bitpos%8);
          uint32_t v=((uint32_t)gl[byte]<<16)|((uint32_t)gl[byte+1]<<8)|((uint32_t)gl[byte+2]);
          idx[k]=(v>>(13-off))&2047;   /* field starts at window bit (23-off), 11 bits down to (13-off) */
          bitpos+=11;
      } }
    for(int oct=0;oct<32;oct++){
        const float *c1=cb+(size_t)idx[oct]*8, *c2=cb+(size_t)idx[32+oct]*8;
        float temp[8];
        for(int t=0;t<8;t++) temp[t]=s1*c1[t]+s2*c2[t];
        fwht8(temp, out+oct*8);
    }
}

/* 132B doctrine-law encoder (16-bit indices) — for parity testing vs 92B */
static void enc_glyph_132c(const float *w, const float *cb, uint8_t *out){
    uint16_t i1s[32], i2s[32];
    float rot_all[32][8];
    for(int g=0;g<32;g++){
        float vec[8]; memcpy(vec,w+g*8,8*sizeof(float));
        fwht8(vec, rot_all[g]);
        float norm=0.f; for(int t=0;t<8;t++) norm+=rot_all[g][t]*rot_all[g][t];
        norm=sqrtf(norm); if(norm<1e-12f) norm=1e-12f;
        float u[8]; for(int t=0;t<8;t++) u[t]=rot_all[g][t]/norm;
        int bi=0; float bd=-1e30f;
        for(int c=0;c<2048;c++){
            const float *cv=cb+(size_t)c*8;
            float d=cv[0]*u[0]+cv[1]*u[1]+cv[2]*u[2]+cv[3]*u[3]+cv[4]*u[4]+cv[5]*u[5]+cv[6]*u[6]+cv[7]*u[7];
            if(d>bd){bd=d;bi=c;}
        }
        i1s[g]=(uint16_t)bi;
        const float *r1=cb+(size_t)bi*8;
        float res[8]; for(int t=0;t<8;t++) res[t]=rot_all[g][t]-r1[t]*norm;
        float rn=0.f; for(int t=0;t<8;t++) rn+=res[t]*res[t]; rn=sqrtf(rn); if(rn<1e-12f) rn=1e-12f;
        float ru[8]; for(int t=0;t<8;t++) ru[t]=res[t]/rn;
        int bi2=0; float bd2=-1e30f;
        for(int c=0;c<2048;c++){
            const float *cv=cb+(size_t)c*8;
            float d=cv[0]*ru[0]+cv[1]*ru[1]+cv[2]*ru[2]+cv[3]*ru[3]+cv[4]*ru[4]+cv[5]*ru[5]+cv[6]*ru[6]+cv[7]*ru[7];
            if(d>bd2){bd2=d;bi2=c;}
        }
        i2s[g]=(uint16_t)bi2;
    }
    double num1=0,num2=0,den=0;
    for(int g=0;g<32;g++){
        const float *r1=cb+(size_t)i1s[g]*8, *r2=cb+(size_t)i2s[g]*8;
        for(int t=0;t<8;t++){ num1+=r1[t]*rot_all[g][t]; num2+=r2[t]*rot_all[g][t]; den+=(r1[t]+r2[t])*(r1[t]+r2[t]); }
    }
    float s1=(float)(num1/(den>1e-12?den:1e-12)), s2=(float)(num2/(den>1e-12?den:1e-12));
    uint16_t hs=fp32_to_fp16(s1), hs2=fp32_to_fp16(s2);
    memcpy(out,&hs,2); memcpy(out+2,&hs2,2);
    for(int g=0;g<32;g++){ uint16_t v=i1s[g]; memcpy(out+4+g*2,&v,2); }
    for(int g=0;g<32;g++){ uint16_t v=i2s[g]; memcpy(out+68+g*2,&v,2); }
}

/* ---------- glyph encoders (new stream) ---------- */
/* 36B median law (encode_glyph_36) */
/* quickselect median-of-256 (O(n) instead of O(n^2) insertion sort) */
static float kth_smallest(float *a, int n, int k){
    int lo=0, hi=n-1;
    while(lo<hi){
        float pivot=a[(lo+hi)/2];
        int i=lo, j=hi;
        while(i<=j){
            while(a[i]<pivot) i++;
            while(a[j]>pivot) j--;
            if(i<=j){ float t=a[i]; a[i]=a[j]; a[j]=t; i++; j--; }
        }
        if(k<=j) hi=j; else if(k>=i) lo=i; else break;
    }
    return a[k];
}
static void enc_glyph_36(const float *w, uint8_t *out){
    float s[256]; memcpy(s,w,256*sizeof(float));
    float m1=kth_smallest(s,256,127);
    float m2=kth_smallest(s,256,128);
    float median=(m1+m2)*0.5f;
    float amax=0.f;
    for(int i=0;i<256;i++){ float c=fabsf(w[i]-median); if(c>amax)amax=c; }
    float scale=amax>0?amax:1.f;
    uint16_t hs=fp32_to_fp16(scale), hb=fp32_to_fp16(median);
    memcpy(out,&hs,2); memcpy(out+2,&hb,2);
    memset(out+4,0,32);
    for(int i=0;i<256;i++) if(w[i]-median>=0.f) out[4+(i>>3)]|=(uint8_t)(0x80>>(i&7));
}
/* 36B variant: optimal 2-means centroids (same 36B container: bias +- scale/2).
   Strictly better least-squares fit than median+-max on skewed glyphs. */
static void enc_glyph_36_k2(const float *w, uint8_t *out){
    float c0=w[0], c1=w[0];
    for(int i=1;i<256;i++){ if(w[i]<c0)c0=w[i]; if(w[i]>c1)c1=w[i]; }
    for(int it=0; it<8; it++){
        double s0=0,s1=0; int n0=0,n1=0;
        for(int i=0;i<256;i++){ float d0=fabsf(w[i]-c0), d1=fabsf(w[i]-c1);
            if(d0<=d1){ s0+=w[i]; n0++; } else { s1+=w[i]; n1++; } }
        float nc0 = n0? (float)(s0/n0) : c0;
        float nc1 = n1? (float)(s1/n1) : c1;
        if(nc0==c0 && nc1==c1) break;
        c0=nc0; c1=nc1;
    }
    float bias=(c0+c1)*0.5f, scale=fabsf(c1-c0)*0.5f;
    uint16_t hs=fp32_to_fp16(scale), hb=fp32_to_fp16(bias);
    memcpy(out,&hs,2); memcpy(out+2,&hb,2);
    memset(out+4,0,32);
    for(int i=0;i<256;i++){ float d0=fabsf(w[i]-c0), d1=fabsf(w[i]-c1); if(d1<d0) out[4+(i>>3)]|=(uint8_t)(0x80>>(i&7)); }
}

/* 92B packed 11-bit dual-stage RVQ (encode_glyph_132 law + 11-bit packing) */
static void enc_glyph_92(const float *w, const float *cb, uint8_t *out){
    uint16_t i1s[32], i2s[32];
    float rot_all[32][8];
    for(int g=0;g<32;g++){
        float vec[8]; memcpy(vec,w+g*8,8*sizeof(float));
        fwht8(vec, rot_all[g]);
        float norm=0.f; for(int t=0;t<8;t++) norm+=rot_all[g][t]*rot_all[g][t];
        norm=sqrtf(norm); if(norm<1e-12f) norm=1e-12f;
        float u[8]; for(int t=0;t<8;t++) u[t]=rot_all[g][t]/norm;
        int bi=0; float bd=-1e30f;
        for(int c=0;c<2048;c++){
            const float *cv=cb+(size_t)c*8;
            float d=cv[0]*u[0]+cv[1]*u[1]+cv[2]*u[2]+cv[3]*u[3]+cv[4]*u[4]+cv[5]*u[5]+cv[6]*u[6]+cv[7]*u[7];
            if(d>bd){bd=d;bi=c;}
        }
        i1s[g]=(uint16_t)bi;
        const float *r1=cb+(size_t)bi*8;
        float res[8]; for(int t=0;t<8;t++) res[t]=rot_all[g][t]-r1[t]*norm;
        float rn=0.f; for(int t=0;t<8;t++) rn+=res[t]*res[t]; rn=sqrtf(rn); if(rn<1e-12f) rn=1e-12f;
        float ru[8]; for(int t=0;t<8;t++) ru[t]=res[t]/rn;
        int bi2=0; float bd2=-1e30f;
        for(int c=0;c<2048;c++){
            const float *cv=cb+(size_t)c*8;
            float d=cv[0]*ru[0]+cv[1]*ru[1]+cv[2]*ru[2]+cv[3]*ru[3]+cv[4]*ru[4]+cv[5]*ru[5]+cv[6]*ru[6]+cv[7]*ru[7];
            if(d>bd2){bd2=d;bi2=c;}
        }
        i2s[g]=(uint16_t)bi2;
    }
    /* global least-squares scales (python law) */
    double num1=0,num2=0,den=0;
    for(int g=0;g<32;g++){
        const float *r1=cb+(size_t)i1s[g]*8, *r2=cb+(size_t)i2s[g]*8;
        for(int t=0;t<8;t++){
            num1+=r1[t]*rot_all[g][t]; num2+=r2[t]*rot_all[g][t];
            den+=(r1[t]+r2[t])*(r1[t]+r2[t]);
        }
    }
    float s1=(float)(num1/(den>1e-12?den:1e-12));
    float s2=(float)(num2/(den>1e-12?den:1e-12));
    uint16_t hs=fp32_to_fp16(s1), hs2=fp32_to_fp16(s2);
    memcpy(out,&hs,2); memcpy(out+2,&hs2,2);
    memset(out+4,0,88);
    long bitpos=0;
    for(int k=0;k<64;k++){
        uint32_t v = k<32 ? i1s[k] : i2s[k-32];
        for(int b=10;b>=0;b--){
            long byte=4+bitpos/8; int off=(int)(bitpos%8);
            if((v>>b)&1) out[byte] |= (uint8_t)(0x80>>off);
            bitpos++;
        }
    }
}

/* lossless transcode: 132B glyph (16-bit idx) -> 92B (11-bit packed) */
static void transcode_132_to_92(const uint8_t *gl, uint8_t *out){
    out[0]=gl[0]; out[1]=gl[1]; out[2]=gl[2]; out[3]=gl[3];
    memset(out+4,0,88);
    long bitpos=0;
    for(int k=0;k<64;k++){
        uint16_t v; memcpy(&v, gl+4+k*2, 2); v&=2047;
        for(int b=10;b>=0;b--){
            long byte=4+bitpos/8; int off=(int)(bitpos%8);
            if((v>>b)&1) out[byte]|=(uint8_t)(0x80>>off);
            bitpos++;
        }
    }
}

/* ---------- layout ---------- */
typedef struct { int layer; char name[160]; char full[256]; int codec_id; int bpg; int d_in,d_out; long glyphs; long tensor_bytes; long off; int new_bpg; double err36; } Proj;

static int jnum(const char *obj, const char *key, long long *out){
    char pat[64]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char *p=strstr(obj,pat); if(!p) return 0;
    p=strchr(p,':'); if(!p) return 0; p++;
    while(*p==' '||*p=='\t') p++;
    if(*p=='-'||(*p>='0'&&*p<='9')){ *out=strtoll(p,NULL,10); return 1; }
    return 0;
}
static void jstr(const char *obj, const char *key, char *out, size_t n){
    char pat[64]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char *p=strstr(obj,pat); out[0]=0; if(!p) return;
    p=strchr(p,':'); if(!p) return; p=strchr(p,'\"'); if(!p) return; p++;
    size_t k=0; while(*p && *p!='\"' && k<n-1) out[k++]=*p++;
    out[k]=0;
}
static Proj *load_layout(const char *path, int *n_out){
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"FATAL: %s\n",path);exit(1);}
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(fsz+1); fread(b,1,fsz,f); b[fsz]=0; fclose(f);
    int cap=512; Proj *P=calloc(cap,sizeof(Proj)); int np=0;
    const char *p=b;
    while((p=strchr(p,'{'))!=NULL){
        /* object must contain a codec marker and an offset */
        const char *obj=p;
        long long bpg=0, off=0, gl=0, din=0, dout=0, lay=0;
        int has_bpg=jnum(obj,"bytes_per_glyph",&bpg);
        int has_off=jnum(obj,"stream_byte_offset",&off);
        if(!has_bpg){ int o2=jnum(obj,"byte_offset",&off); has_off=o2; }
        if(!has_bpg||!has_off){ p++; continue; }
        int has_gl=jnum(obj,"glyphs",&gl);
        if(!has_gl){
            /* flat manifest: shape [d_out,d_in] */
            const char *sh=strstr(obj,"\"shape\"");
            if(sh){ const char *br=strchr(sh,'['); long a=0,c=0;
                   if(br){ char *e2=NULL; a=strtol(br+1,&e2,10); if(e2&&*e2==',') c=strtol(e2+1,NULL,10); }
                   dout=a; din=c; gl=dout*((din+255)/256); has_gl=1; }
        }
        if(!has_gl){ p++; continue; }
        jnum(obj,"layer",&lay);
        jstr(obj,"tensor_name",P[np].name,sizeof P[np].name);
        jstr(obj,"full_key",P[np].full,sizeof P[np].full);
        P[np].layer=(int)lay; P[np].codec_id=(int)bpg; P[np].bpg=(int)bpg;
        P[np].d_in=(int)din; P[np].d_out=(int)dout;
        P[np].glyphs=(long)gl; P[np].off=(long)off;
        np++; if(np>=cap){ cap*=2; P=realloc(P,cap*sizeof(Proj)); }
        p++;
    }
    *n_out=np; return P;
}

static uint8_t *mmap_file(const char *path, long *size){
    int fd=open(path,O_RDONLY); if(fd<0){fprintf(stderr,"FATAL: open %s\n",path);exit(1);}
    struct stat st; fstat(fd,&st); *size=st.st_size;
    uint8_t *m=mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(m==MAP_FAILED){fprintf(stderr,"FATAL: mmap %s\n",path);exit(1);}
    return m;
}

/* decode one glyph of the ORIGINAL stream to 256 fp32 */
static void dec_orig_glyph(const uint8_t *gl, int bpg, const float *cb, float *w){
    if(bpg==132) dec_glyph_132(gl,cb,w);
    else if(bpg==48){ float tmp[256]; dec_glyph_48(gl,NULL,tmp,NULL); memcpy(w,tmp,256*sizeof(float)); }
    else if(bpg==36) dec_glyph_36(gl,w);
    else { fprintf(stderr,"FATAL: unknown orig bpg %d\n",bpg); exit(1); }
}

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: reencode31b encode|verify ...\n"); return 1; }
    if(!strcmp(argv[1],"encode")){
        if(argc<7){ fprintf(stderr,"usage: reencode31b encode <stream> <layout.json> <codebook.bin> <out_base> [budget]\n"); return 1; }
        const char *stream_path=argv[2], *layout_path=argv[3], *cb_path=argv[4], *out_base=argv[5];
        double budget = argc>6 ? atof(argv[6]) : 4.76e9;
        long ssize; uint8_t *S=mmap_file(stream_path,&ssize);
        long cbsize; uint8_t *CBraw=mmap_file(cb_path,&cbsize);
        float *CB=(float*)CBraw;
        int n; Proj *P=load_layout(layout_path,&n);
        printf("loaded %d projections, stream %.2f GB, budget %.2f GB\n", n, ssize/1e9, budget/1e9);

        /* pass 1: per-projection err36 (requant 36B error vs orig decode) */
        float w[256], w36[256];
        { char cp[600]; snprintf(cp,sizeof cp,"%s.err36",out_base);
          FILE *cf=fopen(cp,"r");
          if(cf){ int pi2; double e2; long done[512]={0};
            while(fscanf(cf,"%d %lf",&pi2,&e2)==2){ if(pi2>=0&&pi2<n){ P[pi2].err36=e2; done[pi2]=1; } }
            fclose(cf);
            int ndone=0; for(int i=0;i<n;i++) if(done[i]) ndone++;
            printf("checkpoint: %d/%d projections already measured\n", ndone, n);
            /* skip measured ones in pass 1 */
            #pragma omp parallel for schedule(dynamic)
            for(int pi=0; pi<n; pi++){
                if(done[pi]) continue;
                Proj *p=&P[pi];
                int gpr=(p->d_in+255)/256;
                double err=0;
                for(long r=0;r<p->d_out;r++){
                    for(int g=0;g<gpr;g++){
                        const uint8_t *gl=S+p->off+(r*gpr+g)*p->bpg;
                        dec_orig_glyph(gl,p->bpg,CB,w);
                        uint8_t e36[36]; enc_glyph_36(w,e36);
                        dec_glyph_36(e36,w36);
                        for(int i=0;i<256;i++){ double d=w[i]-w36[i]; err+=d*d; }
                    }
                }
                p->err36=err;
                { char cp2[600]; snprintf(cp2,sizeof cp2,"%s.err36",out_base);
                  FILE *cf2=fopen(cp2,"a"); if(cf2){ fprintf(cf2,"%d %.17g\n",pi,err); fclose(cf2); } }
                if(pi%40==0){ printf("  proj %d %s err36=%.4g\n",pi,p->name,err); fflush(stdout); }
            }
            goto alloc;
          } }
        #pragma omp parallel for schedule(dynamic)
        for(int pi=0; pi<n; pi++){
            Proj *p=&P[pi];
            int gpr=(p->d_in+255)/256;
            double err=0;
            (void)0;
            for(long r=0;r<p->d_out;r++){
                for(int g=0;g<gpr;g++){
                    const uint8_t *gl=S+p->off+(r*gpr+g)*p->bpg;
                    dec_orig_glyph(gl,p->bpg,CB,w);
                    uint8_t e36[36]; enc_glyph_36(w,e36);
                    dec_glyph_36(e36,w36);
                    for(int i=0;i<256;i++){ double d=w[i]-w36[i]; err+=d*d; }
                }
            }
            p->err36=err;
            { char cp[600]; snprintf(cp,sizeof cp,"%s.err36",out_base);
              FILE *cf=fopen(cp,"a"); if(cf){ fprintf(cf,"%d %.17g\n",pi,err); fclose(cf); } }
            if(pi%40==0){ printf("  proj %d %s err36=%.4g\n",pi,p->name,err); fflush(stdout); }
        }
        alloc:
        /* allocation: sort by err36/glyph desc, give 92B while budget allows */
        int *ord=malloc(n*sizeof(int));
        for(int i=0;i<n;i++) ord[i]=i;
        for(int i=1;i<n;i++){ int k=ord[i]; int j=i-1; while(j>=0 && P[ord[j]].err36/P[ord[j]].glyphs < P[k].err36/P[k].glyphs){ ord[j+1]=ord[j]; j--; } ord[j+1]=k; }
        long total_glyphs=0; for(int i=0;i<n;i++) total_glyphs+=P[i].glyphs;
        double base=16.0+(double)total_glyphs*36.0;      /* all-36B baseline */
        double rem=budget-base;                            /* budget for 92B upgrades */
        double used=base;
        for(int i=0;i<n;i++){ Proj *p=&P[ord[i]];
            double up=p->glyphs*56.0;                      /* 92B instead of 36B */
            if(rem>=up){ p->new_bpg=92; rem-=up; used+=up; }
            else p->new_bpg=36;
        }
        int n92=0; for(int i=0;i<n;i++) if(P[i].new_bpg==92) n92++;
        printf("allocation: %d projections at 92B, %d at 36B, total %.3f GB (budget %.2f)\n", n92, n-n92, used/1e9, budget/1e9);

        /* pass 2: write new stream */
        char path[512];
        snprintf(path,sizeof path,"%s.lanu",out_base);
        FILE *O=fopen(path,"wb"); if(!O){fprintf(stderr,"FATAL: %s\n",path);return 1;}
        setvbuf(O,NULL,_IOFBF,32*1024*1024);   /* big buffer: SD cards die on 36B writes */
        LanuHeader hdr={LANU_MAGIC,2,0,0,0,0};
        hdr.glyph_count=(uint32_t)total_glyphs;
        fwrite(&hdr,sizeof hdr,1,O);
        long woff=16;
        for(int pi=0; pi<n; pi++){
            Proj *p=&P[pi];
            int gpr=(p->d_in+255)/256;
            long poff=woff;
            /* encode this projection's glyphs in parallel, then one sequential write */
            long nbytes=(long)p->glyphs*p->new_bpg;
            uint8_t *obuf=malloc(nbytes);
            #pragma omp parallel for schedule(static)
            for(long gi=0; gi<p->glyphs; gi++){
                long r=gi/gpr, g=(int)(gi%gpr);
                const uint8_t *gl=S+p->off+gi*p->bpg;
                uint8_t *dst=obuf+gi*p->new_bpg;
                if(p->new_bpg==92 && p->bpg==132){ transcode_132_to_92(gl,dst); }
                else {
                    float wl[256];
                    dec_orig_glyph(gl,p->bpg,CB,wl);
                    if(p->new_bpg==92) enc_glyph_92(wl,CB,dst);
                    else enc_glyph_36(wl,dst);
                }
            }
            fwrite(obuf,1,nbytes,O);
            free(obuf);
            woff += p->glyphs*p->new_bpg;
            p->off=poff; p->tensor_bytes=p->glyphs*p->new_bpg; p->bpg=p->new_bpg;
            if(pi%40==0){ printf("  wrote proj %d %s (%dB)\n",pi,p->name,p->new_bpg); fflush(stdout); }
        }
        fclose(O);
        printf("stream written: %s.lanu (%.3f GB)\n", out_base, woff/1e9);
        /* new layout json */
        snprintf(path,sizeof path,"%s.layout.json",out_base);
        FILE *L=fopen(path,"w");
        fprintf(L,"{\"model\":\"gemma4-31b-ramfit\",\"stream_file\":\"%s.lanu\",\"stream_size_bytes\":%ld,\"total_projections\":%d,\"total_macro_glyphs\":%ld,\"projections\":[\n", out_base, woff, n, total_glyphs);
        for(int i=0;i<n;i++){
            Proj *p=&P[i];
            fprintf(L,"{\"layer\":%d,\"tensor_name\":\"%s\",\"full_key\":\"%s\",\"codec\":\"%s\",\"codec_id\":%d,\"bytes_per_glyph\":%d,\"d_in\":%d,\"d_out\":%d,\"glyphs\":%ld,\"tensor_bytes\":%ld,\"stream_byte_offset\":%ld}%s\n",
                p->layer,p->name,p->full, p->new_bpg==92?"OPT4P":"OPT5", p->new_bpg==92?4:5, p->new_bpg, p->d_in,p->d_out,p->glyphs,p->tensor_bytes,p->off, i+1<n?",":"");
        }
        fprintf(L,"]}\n");
        fclose(L);
        printf("layout written: %s.layout.json\n", out_base);
        return 0;
    }
    if(!strcmp(argv[1],"selftest")){
        /* decode real 132B glyphs from the stream, re-encode 92B, decode, compare */
        if(argc<6){ fprintf(stderr,"usage: reencode31b selftest <stream> <layout.json> <codebook.bin> [nglyphs]\n"); return 1; }
        int ng = argc>5?atoi(argv[5]):8;
        long ss; uint8_t *S=mmap_file(argv[2],&ss);
        long cs; uint8_t *CBraw=mmap_file(argv[4],&cs);
        float *CB=(float*)CBraw;
        int n; Proj *P=load_layout(argv[3],&n);
        Proj *p=&P[0];
        int gpr=(p->d_in+255)/256;
        float w[256],w92[256];
        double maxd=0, rel=0;
        for(int k=0;k<ng;k++){
            const uint8_t *gl=S+p->off+(long)k*p->bpg;
            dec_orig_glyph(gl,p->bpg,CB,w);
            uint8_t e[92]; enc_glyph_92(w,CB,e);
            dec_glyph_92(e,CB,w92);
            double d=0,nrm=0;
            for(int i=0;i<256;i++){ double dd=w[i]-w92[i]; d+=dd*dd; nrm+=w[i]*w[i]; }
            if(d>maxd) maxd=d; rel+=d;
            printf("glyph %d: max|d|=%.3e rel=%.3e\n", k, sqrt(d/256), sqrt(d/(nrm+1e-12)));
        }
        printf("92B round-trip vs 132B decode: max RMS diff %.3e\n", sqrt(maxd/256));
        /* parity: enc_glyph_132c vs enc_glyph_92 on the SAME weights must decode identically */
        { const uint8_t *gl=S+p->off;
          dec_orig_glyph(gl,p->bpg,CB,w);
          uint8_t e132[132], e92[92];
          enc_glyph_132c(w,CB,e132); enc_glyph_92(w,CB,e92);
          float d132[256], d92[256];
          dec_glyph_132(e132,CB,d132); dec_glyph_92(e92,CB,d92);
          double md=0; for(int i=0;i<256;i++){ double dd=d132[i]-d92[i]; if(fabs(dd)>md) md=fabs(dd); }
          printf("132c-vs-92 decode parity on same weights: max|d| = %.3e (must be 0)\n", md);
        }
        return 0;
    }
    if(!strcmp(argv[1],"cosprobe")){
        /* cosprobe <stream> <layout> <codebook> [nproj] : compare 36B median law vs 2-means law */
        if(argc<6){ fprintf(stderr,"usage: reencode31b cosprobe <stream> <layout> <codebook> [nproj]\n"); return 1; }
        int np_=argc>5?atoi(argv[5]):5;
        long ss; uint8_t *S=mmap_file(argv[2],&ss);
        long cs; uint8_t *CBraw=mmap_file(argv[4],&cs);
        float *CB=(float*)CBraw;
        int n; Proj *P=load_layout(argv[3],&n);
        double sum_med=0,sum_k2=0; int cnt=0;
        for(int i=0;i<n && cnt<np_;i++){
            if(P[i].bpg!=48) continue;          /* compare on the 48B middle lane */
            Proj *p=&P[i];
            int gpr=(p->d_in+255)/256;
            double dot_m=0,dot_k=0,nm=0,n1=0,n2=0;
            float w[256],wd[256];
            long rows=p->d_out; if(rows>32) rows=32;
            for(long r=0;r<rows;r++)
                for(int g=0;g<gpr;g++){
                    const uint8_t *gl=S+p->off+(r*gpr+g)*p->bpg;
                    dec_orig_glyph(gl,p->bpg,CB,w);
                    uint8_t e1[36],e2[36];
                    enc_glyph_36(w,e1); enc_glyph_36_k2(w,e2);
                    dec_glyph_36(e1,wd);
                    for(int t=0;t<256;t++){ dot_m+=w[t]*wd[t]; nm+=w[t]*w[t]; n1+=wd[t]*wd[t]; }
                    dec_glyph_36(e2,wd);
                    for(int t=0;t<256;t++){ dot_k+=w[t]*wd[t]; n2+=wd[t]*wd[t]; }
                }
            double cos_m=(nm>0&&n1>0)?dot_m/sqrt(nm*n1):0, cos_k=(nm>0&&n2>0)?dot_k/sqrt(nm*n2):0;
            printf("proj %d %-34s median-law cos=%.5f  2-means cos=%.5f\n",i,p->name,cos_m,cos_k);
            sum_med+=cos_m; sum_k2+=cos_k; cnt++;
        }
        printf("MEAN over %d projections: median-law %.5f | 2-means %.5f\n",cnt,sum_med/cnt,sum_k2/cnt);
        return 0;
    }
    if(!strcmp(argv[1],"verify")){
        if(argc<7){ fprintf(stderr,"usage: reencode31b verify <orig_stream> <orig_layout> <new_stream> <new_layout> [max_proj]\n"); return 1; }
        int maxp = argc>7?atoi(argv[7]):20;
        long s1; uint8_t *S1=mmap_file(argv[2],&s1);
        long s2; uint8_t *S2=mmap_file(argv[4],&s2);
        long cbs; uint8_t *CBraw=mmap_file("/storage/emulated/0/amara-models/glyph-stream-31b/s7_codebook_2048x8.bin",&cbs);
        float *CB=(float*)CBraw;
        int n1,n2; Proj *P1=load_layout(argv[3],&n1); Proj *P2=load_layout(argv[5],&n2);
        if(n1!=n2){ fprintf(stderr,"FATAL: proj count mismatch %d vs %d\n",n1,n2); return 1; }
        printf("%-46s %5s %5s %8s\n","projection","bpg","new","cos");
        double worst=1;
        for(int i=0;i<n1 && i<maxp;i++){
            Proj *a=&P1[i], *b=&P2[i];
            int gpr=(a->d_in+255)/256;
            double dot=0,n1v=0,n2v=0;
            float w1[256],w2[256];
            long rows=a->d_out; if(rows>64) rows=64;   /* sample 64 rows */
            for(long r=0;r<rows;r++)
                for(int g=0;g<gpr;g++){
                    dec_orig_glyph(S1+a->off+(r*gpr+g)*a->bpg, a->bpg, CB, w1);
                    const uint8_t *gl2=S2+b->off+(r*gpr+g)*b->bpg;
                    if(b->bpg==92) dec_glyph_92(gl2,CB,w2); else dec_glyph_36(gl2,w2);
                    for(int t=0;t<256;t++){ dot+=w1[t]*w2[t]; n1v+=w1[t]*w1[t]; n2v+=w2[t]*w2[t]; }
                }
            double cos = (n1v>0&&n2v>0)? dot/sqrt(n1v*n2v) : 0;
            if(cos<worst) worst=cos;
            printf("%-46s %5d %5d %8.5f\n", a->name, a->bpg, b->bpg, cos);
        }
        printf("worst cos over sampled projections: %.5f\n", worst);
        return 0;
    }
    fprintf(stderr,"unknown mode %s\n",argv[1]);
    return 1;
}
