/* shuffle_stream.c — null control: permute glyph order WITHIN each projection.
 * Same bytes, same container, same codec, same offsets, same runtime — only the
 * mapping from row-position to weight-glyph is destroyed. If a model still
 * "knows" anything after this, the knowledge came from the container or a
 * lookup, not from the learned glyph weights.
 *
 * usage: shuffle_stream <in.lanu> <layout.json> <out.lanu> [seed]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

typedef struct { char name[160]; int bpg; long d_in,d_out; long glyphs; long off; } Proj;

int main(int argc, char **argv){
    if(argc<4){ fprintf(stderr,"usage: shuffle_stream <in.lanu> <layout.json> <out.lanu> [seed]\n"); return 1; }
    /* minimal layout scan: we only need bytes_per_glyph + glyphs + stream_byte_offset
       per projection, in file order. Parse with a tiny hand-rolled scanner. */
    FILE *lf=fopen(argv[2],"rb"); if(!lf){fprintf(stderr,"FATAL: %s\n",argv[2]);return 1;}
    fseek(lf,0,SEEK_END); long lsz=ftell(lf); fseek(lf,0,SEEK_SET);
    char *lb=malloc(lsz+1); fread(lb,1,lsz,lf); lb[lsz]=0; fclose(lf);
    /* find each projection object: locate "bytes_per_glyph": N ... "glyphs": N ... "stream_byte_offset": N */
    int cap=512; Proj *P=calloc(cap,sizeof(Proj)); int np=0;
    int mode=0;   /* 0 = layout json (projections array), 1 = flat manifest map */
    char *cur=NULL;
    if(strstr(lb,"\"projections\"")){ mode=0; cur=strstr(lb,"\"projections\""); }
    else if(strstr(lb,"\"bytes_per_glyph\"")){ mode=1; cur=lb; }
    else { fprintf(stderr,"FATAL: unrecognised layout/manifest format\n"); return 1; }
    while((cur=strstr(cur,"{"))!=NULL){
        char *bpg=strstr(cur,"\"bytes_per_glyph\"");
        char *gl=NULL, *off=NULL, *shape=NULL;
        if(mode==0){
            gl=strstr(cur,"\"glyphs\""); off=strstr(cur,"\"stream_byte_offset\"");
        } else {
            off=strstr(cur,"\"byte_offset\""); shape=strstr(cur,"\"shape\"");
        }
        if(!bpg||!off) break;
        if(mode==0 && (!gl || gl-bpg>600 || off-gl>600)){ cur++; continue; }
        if(mode==1 && (!shape || shape-bpg>400 || off-shape>400)){ cur++; continue; }
        if(np>=cap){ cap*=2; P=realloc(P,cap*sizeof(Proj)); }
        int b=0; long g=0,o=0,d_in=0,d_out=0;
        sscanf(strchr(bpg,':')+1,"%d",&b);
        sscanf(strchr(off,':')+1,"%ld",&o);
        if(mode==0){ sscanf(strchr(gl,':')+1,"%ld",&g); }
        else {
            /* shape: [d_out, d_in] in numpy order */
            long sh[2]={0,0}; char *br=strchr(shape,'[');
            if(br) sscanf(br+1,"%ld,%ld",&sh[0],&sh[1]);
            d_out=sh[0]; d_in=sh[1];
            g=d_out*((d_in+255)/256);
        }
        P[np].bpg=b; P[np].glyphs=g; P[np].off=o; np++;
        cur=off;
    }
    fprintf(stderr,"parsed %d projections from layout\n",np);
    FILE *in=fopen(argv[1],"rb"); if(!in){fprintf(stderr,"FATAL: %s\n",argv[1]);return 1;}
    FILE *out=fopen(argv[3],"wb"); if(!out){fprintf(stderr,"FATAL: %s\n",argv[3]);return 1;}
    unsigned seed = argc>4?(unsigned)atoi(argv[4]):12345u;
    /* copy header + any bytes before the first projection */
    long first=P[0].off;
    { unsigned char hdr[16]; fread(hdr,1,16,in); fwrite(hdr,1,16,out);
      unsigned char *gap=malloc(first-16>0?first-16:1);
      if(first>16){ fread(gap,1,first-16,in); fwrite(gap,1,first-16,out); } free(gap); }
    /* sort projections by stream offset, then walk the stream sequentially:
       copy gaps verbatim, shuffle only non-overlapping glyph regions. */
    for(int i=1;i<np;i++){ Proj key=P[i]; int j=i-1;
        while(j>=0 && P[j].off>key.off){ P[j+1]=P[j]; j--; } P[j+1]=key; }
    long pos=first; long copied=16;
    for(int i=0;i<np;i++){
        Proj *p=&P[i];
        long ng=p->glyphs, bpg=p->bpg;
        long start=p->off, end=p->off+ng*bpg;
        if(start < pos){ continue; }            /* aliased/overlapping entry: skip */
        /* copy the gap verbatim */
        if(start>copied){
            unsigned char *gap=malloc(start-copied);
            fread(gap,1,start-copied,in); fwrite(gap,1,start-copied,out);
            free(gap); copied=start;
        }
        unsigned char *buf=malloc((size_t)ng*bpg);
        fread(buf,1,(size_t)ng*bpg,in);
        long *perm=malloc(ng*sizeof(long));
        for(long k=0;k<ng;k++) perm[k]=k;
        srand(seed+i*7919);
        for(long k=ng-1;k>0;k--){ long j=rand()%(k+1); long t=perm[k]; perm[k]=perm[j]; perm[j]=t; }
        for(long k=0;k<ng;k++) fwrite(buf+(size_t)perm[k]*bpg,1,bpg,out);
        free(buf); free(perm);
        copied=end; pos=end;
    }
    /* copy the tail verbatim */
    { long tail=0; fseek(in,0,SEEK_END); tail=ftell(in)-copied;
      if(tail>0){ unsigned char *tb=malloc(tail); fseek(in,copied,SEEK_SET); fread(tb,1,tail,in); fwrite(tb,1,tail,out); free(tb); } }
    fclose(in); fclose(out);
    fprintf(stderr,"shuffled stream written: %s (seed %u)\n",argv[3],seed);
    return 0;
}
