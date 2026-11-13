#define SEMANTIC_TEST
#include "semantic.cpp"
#include <array>
#include <cstring>
#include <fstream>
#include <cmath>
#include <openssl/sha.h>
using Bytes=std::vector<unsigned char>;
Bytes read(const char*p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open failed");return Bytes(std::istreambuf_iterator<char>(f),{});}
uint64_t get(const Bytes&b,size_t p,int n){if(p+n>b.size())throw std::runtime_error("truncated");uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t(b[p+i])<<(8*i);return v;}
void put(Bytes&b,size_t p,uint64_t v,int n){if(p+n>b.size())throw std::runtime_error("write bounds");for(int i=0;i<n;i++)b[p+i]=(v>>(8*i))&255;}
std::array<unsigned char,32> hash(const unsigned char*p,size_t n){std::array<unsigned char,32>d;SHA256(p,n,d.data());return d;}
struct Bank {Bytes bytes;std::vector<size_t> instructions;};
Bank load(const char* path){
 Bank bank{read(path),{}};auto&b=bank.bytes;
 if(b.size()<108||b.size()>1000000||std::memcmp(b.data(),"LUJ1",4)||get(b,4,1)!=1||get(b,5,1)!=18||get(b,6,2))throw std::runtime_error("invalid LUJ1");
 auto digest=hash(b.data()+108,b.size()-108);
 if(std::memcmp(digest.data(),b.data()+76,32))throw std::runtime_error("body hash");
 auto count=get(b,8,4);if(count!=5)throw std::runtime_error("require five-record planner bank");
 Bytes recovered;size_t p=108;
 const I blueprint[]={32,32,8,1,2};
 for(unsigned i=0;i<count;i++){
  auto nt=get(b,p+5,1),nw=get(b,p+8,4),nb=get(b,p+12,4),ni=get(b,p+16,4);
  if(get(b,p,4)!=i||get(b,p+4,1)!=1||get(b,p+6,2)||nw!=144||ni!=22||(nt!=20&&nt!=21)||nb!=(nt*18+7)/8)throw std::runtime_error("unsupported record");
  p+=20;if(p+nw+nb+ni>b.size())throw std::runtime_error("truncated lanes");
  // QAT4 group-32 BF16 scales and BF16-rounded products, exactly as LUJ1.
  for(unsigned j=0;j<256;j++){
   uint32_t bits=uint32_t(get(b,p+128+(j/32)*2,2))<<16;float scale;std::memcpy(&scale,&bits,4);
   if(!std::isfinite(scale)||scale<0)throw std::runtime_error("invalid scale");
   int code=((b[p+j/2]>>((j%2)*4))&15)-8;float value=code*scale;
   if(!std::isfinite(value))throw std::runtime_error("nonfinite weight");
   std::memcpy(&bits,&value,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
   std::memcpy(&value,&bits,4);if(!std::isfinite(value))throw std::runtime_error("rounding overflow");
   for(int k=0;k<4;k++)recovered.push_back((bits>>(8*k))&255);
  }
  if(nt*18%8 && (b[p+nw+nb-1]>>(nt*18%8))!=0)throw std::runtime_error("token padding");
  p+=nw+nb;
  if(get(b,p,1)!=1||get(b,p+1,1)||get(b,p+2,2)!=65535||get(b,p+4,2)!=65535||get(b,p+6,8)!=uint64_t(blueprint[i])||get(b,p+14,8))throw std::runtime_error("model blueprint mismatch");
  bank.instructions.push_back(p);p+=22;
 }
 if(p!=b.size())throw std::runtime_error("trailing bytes");
 digest=hash(recovered.data(),recovered.size());if(std::memcmp(digest.data(),b.data()+44,32))throw std::runtime_error("weight hash");
 return bank;
}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: glyph_bridge planner.luj 'Start with ...' output.luj");
 auto bank=load(argv[1]);auto result=solve(argv[2]);auto&b=bank.bytes;
 std::string q=argv[2];auto pos=q.find(';');auto first=trim(q.substr(0,pos));I initial=integer(first.substr(11));
 const std::vector<std::string> names={"", "add","subtract","multiply","","","divide","minimum","maximum","power","remainder"};
 I state=initial;
 for(size_t i=0;i<5;i++){
  Step s=i<result.steps.size()?result.steps[i]:Step{"add",0,false};
  size_t p=bank.instructions[i];auto op=std::find(names.begin(),names.end(),s.op)-names.begin();
  put(b,p,op,1);put(b,p+1,0,1);
  int ar=i?int(i-1):-1,br=-1;I ai=i?0:initial,bi=s.operand;
  if(s.reverse){std::swap(ar,br);std::swap(ai,bi);}
  put(b,p+2,uint16_t(ar),2);put(b,p+4,uint16_t(br),2);put(b,p+6,uint64_t(ai),8);put(b,p+14,uint64_t(bi),8);
  state=execute(state,s);
 }
 std::vector<I> decoded;
 for(size_t p:bank.instructions){
  auto operand=[&](size_t refpos,size_t immpos){uint16_t ref=get(b,refpos,2);uint64_t raw=get(b,immpos,8);I imm;std::memcpy(&imm,&raw,8);if(ref==65535)return imm;if(ref>=decoded.size()||imm)throw std::runtime_error("invalid dependency");return decoded[ref];};
  auto op=get(b,p,1);if(op>=names.size()||names[op].empty())throw std::runtime_error("bad opcode");
  decoded.push_back(execute(operand(p+2,p+6),Step{names[op],operand(p+4,p+14),false}));
 }
 if(decoded.back()!=state)throw std::runtime_error("serialized instruction mismatch");
 if(state!=result.value)throw std::runtime_error("execution mismatch");
 auto digest=hash(b.data()+108,b.size()-108);std::copy(digest.begin(),digest.end(),b.begin()+76);
 std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(b.data()),b.size());if(!out)throw std::runtime_error("write failed");
 std::cout<<"Validated 1280 decoded weights; compiled "<<result.steps.size()<<" steps into "<<b.size()<<" bytes; result "<<state<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
