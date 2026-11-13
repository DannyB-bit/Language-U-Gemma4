// GBF1: lossless BF16 weight glyph stream. Not the LUJ1 token/program ABI.
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <openssl/sha.h>
#include <zstd.h>
using B=std::vector<unsigned char>;
B read(const char*p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open");return B(std::istreambuf_iterator<char>(f),{});}
uint64_t get(const B&b,size_t p,int n){if(p+n>b.size())throw std::runtime_error("truncated");uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t(b[p+i])<<(8*i);return v;}
void add(B&b,uint64_t v,int n){for(int i=0;i<n;i++)b.push_back((v>>(i*8))&255);}
void save(const char*p,const B&b){std::ofstream f(p,std::ios::binary);f.write((const char*)b.data(),b.size());if(!f)throw std::runtime_error("write");}
std::array<unsigned char,32> hash(const B&b){std::array<unsigned char,32>d;SHA256(b.data(),b.size(),d.data());return d;}
B decode(const B&b){
 if(b.size()<52||std::memcmp(b.data(),"GBF1",4))throw std::runtime_error("magic");
 uint64_t size=get(b,4,8),header=get(b,12,8);
 if(size>2000000000||header>size||header<8||header>b.size()-52||(size-header)%2)throw std::runtime_error("bounds");
 B out(b.begin()+52,b.begin()+52+header);out.reserve(size);size_t p=52+header;
 while(out.size()<size){auto n=get(b,p,2);auto packed=get(b,p+2,2);p+=4;
  auto expected=std::min<uint64_t>(512,size-out.size());
  if(n!=expected||!packed||packed>512||p+packed>b.size())throw std::runtime_error("record bounds");
  B plane(n);if(packed==n)std::copy_n(b.begin()+p,n,plane.begin());else{
   if(ZSTD_findFrameCompressedSize(b.data()+p,packed)!=packed)throw std::runtime_error("frame length");
   auto r=ZSTD_decompress(plane.data(),n,b.data()+p,packed);if(ZSTD_isError(r)||r!=n)throw std::runtime_error("decompress");
  }
  for(size_t i=0;i<n/2;i++){out.push_back(plane[i]);out.push_back(plane[i+n/2]);}p+=packed;
 }
 if(p!=b.size())throw std::runtime_error("trailing bytes");
 auto d=hash(out);if(std::memcmp(d.data(),b.data()+20,32))throw std::runtime_error("source hash");return out;
}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: compile_bf16 pack|unpack input output");
 B in=read(argv[2]);if(std::string(argv[1])=="unpack"){auto out=decode(in);save(argv[3],out);std::cout<<"restored_bytes="<<out.size()<<"\n";return 0;}
 if(std::string(argv[1])!="pack")throw std::runtime_error("mode");
 auto h=get(in,0,8);if(h>10000000||h+8>in.size()||(in.size()-h-8)%2)throw std::runtime_error("safetensors header");
 // Require every stored dtype declaration to be BF16. Preserve JSON verbatim.
 std::string meta((const char*)in.data()+8,h);size_t pos=0,tensors=0;
 while((pos=meta.find("\"dtype\"",pos))!=std::string::npos){auto colon=meta.find(':',pos);auto q=meta.find('"',colon);auto end=meta.find('"',q+1);if(colon==std::string::npos||q==std::string::npos||end==std::string::npos||meta.substr(q+1,end-q-1)!="BF16")throw std::runtime_error("requires BF16 tensors");++tensors;pos=end+1;}
 if(!tensors)throw std::runtime_error("no BF16 tensors");
 B out={'G','B','F','1'};add(out,in.size(),8);add(out,h+8,8);auto d=hash(in);out.insert(out.end(),d.begin(),d.end());out.insert(out.end(),in.begin(),in.begin()+h+8);
 size_t count=0;B zipped(ZSTD_compressBound(512));
 for(size_t p=h+8;p<in.size();p+=512){size_t n=std::min<size_t>(512,in.size()-p);B plane(n);for(size_t i=0;i<n/2;i++){plane[i]=in[p+2*i];plane[i+n/2]=in[p+2*i+1];}auto z=ZSTD_compress(zipped.data(),zipped.size(),plane.data(),n,3);if(ZSTD_isError(z))throw std::runtime_error("compress");add(out,n,2);bool compressed=z<n;add(out,compressed?z:n,2);auto &payload=compressed?zipped:plane;out.insert(out.end(),payload.begin(),payload.begin()+(compressed?z:n));++count;}
 // Independently parse the emitted stream before saving it.
 auto restored=decode(out);if(restored!=in)throw std::runtime_error("round-trip mismatch");save(argv[3],out);
 std::cout<<"tensors="<<tensors<<" weights="<<(in.size()-h-8)/2<<" glyphs="<<count<<" source_bytes="<<in.size()<<" glyph_bytes="<<out.size()<<" exact_roundtrip=PASS\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
