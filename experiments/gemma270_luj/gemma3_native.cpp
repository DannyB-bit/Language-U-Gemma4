// Gemma 3 270M native BF16 reference inference, C++17, CPU-only.
// Uses safetensors fully reconstructed from LUJ1 glyphs; no Python at inference time.
// Architecture follows Gemma3ForCausalLM text module; reference FP32 operations
// are not claimed numerically equivalent to the original BF16 Transformers kernels.
#include <sys/mman.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
using F = std::vector<float>;
static void ck(bool x,const std::string& msg){if(!x)throw std::runtime_error(msg);}
static uint32_t rd32(const unsigned char* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
static uint64_t rd64(const unsigned char* p){return uint64_t(rd32(p))|(uint64_t(rd32(p+4))<<32);}
static float bf16(uint16_t x){uint32_t q=uint32_t(x)<<16;float v;std::memcpy(&v,&q,4);return v;}
struct Tensor {const uint16_t* data=nullptr;int rows=0,cols=0;};
class Model {
 int fd=-1;size_t size=0;unsigned char* m=nullptr;
 std::map<std::string,Tensor> tensors;
public:
 explicit Model(const std::string& path,const std::string& expected_sha){
  fd=open(path.c_str(),O_RDONLY);ck(fd>=0,"cannot open model: "+path);
  struct stat st{};ck(fstat(fd,&st)==0&&st.st_size>27000,"invalid model size");size=size_t(st.st_size);
  m=reinterpret_cast<unsigned char*>(mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0));ck(m!=MAP_FAILED,"mmap failed");
  const uint64_t head=rd64(m);ck(head>0&&head<1000000&&size>=8+head,"invalid safetensors header");
  // Bind this reference runner to the user's exact, SHA-verified checkpoint.
  unsigned char identity[32]{};SHA256(m,size,identity);
  static constexpr char lookup[]="0123456789abcdef";std::string id;
  for(auto byte:identity){id+=lookup[byte>>4];id+=lookup[byte&15];}
  ck(expected_sha.size()==64&&expected_sha.find_first_not_of("0123456789abcdef")==std::string::npos, "invalid expected SHA256");
  ck(id==expected_sha, "unexpected checkpoint SHA256");
  std::string json(reinterpret_cast<char*>(m+8),size_t(head));
  static const std::regex rx(R"LUJ("([^"]+)":\{"dtype":"BF16","shape":\[([^\]]+)\],"data_offsets":\[([0-9]+),([0-9]+)\]\})LUJ");
  for(auto it=std::sregex_iterator(json.begin(),json.end(),rx);it!=std::sregex_iterator();++it){
   const auto a=*it;std::string key=a[1].str(),shape=a[2].str();
   size_t off=std::stoull(a[3].str()),end=std::stoull(a[4].str());
   ck(end>=off&&end<=size-8-head&&((end-off)%2)==0,"tensor out of bounds: "+key);
   std::vector<int> dims;std::string x;
   for(char c:shape){if(c==','){dims.push_back(std::stoi(x));x.clear();}else if(c>='0'&&c<='9')x.push_back(c);}
   if(!x.empty()){dims.push_back(std::stoi(x));}
   ck(!dims.empty()&&dims.size()<=2,"bad dims: "+key);
   const int rows=dims[0],cols=dims.size()==2?dims[1]:1;
   ck(uint64_t(rows)*uint64_t(cols)*2==end-off,"tensor shape/offset mismatch: "+key);
   tensors[key]={reinterpret_cast<const uint16_t*>(m+8+head+off),rows,cols};
  }
  ck(tensors.size()==236,"expected 236 Gemma3 BF16 tensors, got "+std::to_string(tensors.size()));
  ck(at("model.embed_tokens.weight").rows==262144&&at("model.embed_tokens.weight").cols==640,"wrong embedding shape");
 }
 ~Model(){if(m&&m!=MAP_FAILED)munmap(m,size);if(fd>=0)close(fd);}
 Model(const Model&)=delete;Model& operator=(const Model&)=delete;
 const Tensor& at(const std::string& name)const{auto it=tensors.find(name);ck(it!=tensors.end(),"missing "+name);return it->second;}
 F linear(const std::string& name,const F& x)const{
  const auto& t=at(name);ck(int(x.size())==t.cols,"projection shape error "+name);
  F o(size_t(t.rows));
#pragma omp parallel for schedule(static)
  for(int i=0;i<t.rows;i++){
   const uint16_t* row=t.data+size_t(i)*size_t(t.cols);
   float sum=0;
#pragma omp simd reduction(+:sum)
   for(int j=0;j<t.cols;j++)sum+=bf16(row[j])*x[size_t(j)];
   o[size_t(i)]=sum;
  }
  return o;
 }
 F row(const std::string& name,size_t i)const{
  const auto& t=at(name);ck(i<size_t(t.rows),"row out of range");F v(size_t(t.cols));
  for(int j=0;j<t.cols;j++)v[size_t(j)]=bf16(t.data[i*size_t(t.cols)+size_t(j)]);
  return v;
 }
 F rms(const F& x,const std::string& name)const{
  const auto& norm=at(name);ck(norm.cols==1&&norm.rows==int(x.size()),"RMS shape mismatch "+name);
  double sq=0;for(float v:x)sq+=double(v)*double(v);
  const float scale=float(1/std::sqrt(sq/x.size()+1e-6));F out(x.size());
  for(size_t i=0;i<x.size();i++)out[i]=x[i]*scale*(1+bf16(norm.data[i]));
  return out;
 }
};
static void add(F& target,const F& x){ck(target.size()==x.size(),"residual shape mismatch");for(size_t i=0;i<target.size();i++)target[i]+=x[i];}
static F gelu_gate(const F& gate,const F& up){ck(gate.size()==up.size(),"MLP size mismatch");F out(gate.size());
 for(size_t i=0;i<out.size();i++){
  const float v=gate[i],inner=0.7978845608028654f*(v+0.044715f*v*v*v);
  out[i]=0.5f*v*(1.0f+std::tanh(inner))*up[i];
 }return out;
}
static void rotary(F& q,int h,int position,double base){ck(q.size()==size_t(h*256),"RoPE dimension mismatch");
 for(int head=0;head<h;head++)for(int i=0;i<128;i++){
  const double theta=position*std::pow(base,-2.0*i/256.0);
  float c=float(std::cos(theta)),s=float(std::sin(theta));
  int x=head*256+i,y=x+128;float a=q[size_t(x)],b=q[size_t(y)];
  q[size_t(x)]=a*c-b*s;q[size_t(y)]=a*s+b*c;
 }
}
struct KVC{std::vector<F> keys,values;};
class Gemma3 {
 const Model& m;std::array<KVC,18> cache{};int position=0;
 static std::string l(int layer,const std::string& path){return "model.layers."+std::to_string(layer)+"."+path+".weight";}
 public:explicit Gemma3(const Model& model):m(model){}
 F forward(uint32_t token){ck(token<262144,"token out of vocabulary range");
  F state=m.row("model.embed_tokens.weight",token);float factor=std::sqrt(640.0f);for(float& x:state)x*=factor;
  const int pos=position++;
  for(int layer=0;layer<18;layer++){
   bool local=layer%6!=5;double base=local?10000.0:1000000.0;
   F in=m.rms(state,l(layer,"input_layernorm"));
   F q=m.linear(l(layer,"self_attn.q_proj"),in);
   F k=m.linear(l(layer,"self_attn.k_proj"),in);
   F v=m.linear(l(layer,"self_attn.v_proj"),in);
   for(int h=0;h<4;h++){F sub(q.begin()+h*256,q.begin()+(h+1)*256);
    sub=m.rms(sub,l(layer,"self_attn.q_norm"));std::copy(sub.begin(),sub.end(),q.begin()+h*256);}
   k=m.rms(k,l(layer,"self_attn.k_norm"));
   rotary(q,4,pos,base);rotary(k,1,pos,base);
   auto& kv=cache[size_t(layer)];kv.keys.push_back(std::move(k));kv.values.push_back(std::move(v));
   const int first=local?std::max(0,int(kv.keys.size())-512):0;
   F attn(1024,0);
   for(int head=0;head<4;head++){
    const float* qp=q.data()+head*256;
    const size_t n=kv.keys.size()-size_t(first);
    F scores(n);float mx=-std::numeric_limits<float>::infinity();
    for(size_t j=0;j<n;j++){
     const auto& key=kv.keys[size_t(first)+j];float dot=0;
     for(int d=0;d<256;d++)dot+=qp[d]*key[size_t(d)];
     float logit=dot/16.0f;scores[j]=logit;mx=std::max(mx,logit);
    }
    float denom=0;for(float& a:scores){a=std::exp(a-mx);denom+=a;}ck(denom>0&&std::isfinite(denom),"invalid attention normalization");
    for(size_t j=0;j<n;j++){
     float p=scores[j]/denom;const F& val=kv.values[size_t(first)+j];
     for(int d=0;d<256;d++)attn[size_t(head*256+d)]+=p*val[size_t(d)];
    }
   }
   F o=m.linear(l(layer,"self_attn.o_proj"),attn);
   add(state,m.rms(o,l(layer,"post_attention_layernorm")));
   F ffn=m.rms(state,l(layer,"pre_feedforward_layernorm"));
   F gate=m.linear(l(layer,"mlp.gate_proj"),ffn);
   F up=m.linear(l(layer,"mlp.up_proj"),ffn);
   F down=m.linear(l(layer,"mlp.down_proj"),gelu_gate(gate,up));
   add(state,m.rms(down,l(layer,"post_feedforward_layernorm")));
   for(float f:state)ck(std::isfinite(f),"nonfinite model activation at layer "+std::to_string(layer));
  }
  return m.rms(state,"model.norm.weight");
 }
 F logits(const F& x)const{return m.linear("model.embed_tokens.weight",x);}
};
static std::vector<uint32_t> parse_ids(std::string text){std::vector<uint32_t> out;size_t start=0;do{
 size_t end=text.find(',',start);std::string s=text.substr(start,end==std::string::npos?end:end-start);
 ck(!s.empty()&&s.find_first_not_of("0123456789")==std::string::npos,"token ID list must be comma-separated nonnegative decimals");
 uint64_t x=std::stoull(s);ck(x<262144,"token outside Gemma3 vocabulary");out.push_back(uint32_t(x));if(end==std::string::npos)break;start=end+1;
 }while(true);ck(!out.empty()&&out.size()<=64,"accepted token count is 1..64");return out;}
int main(int argc,char** argv){try{
 ck(argc>=4,"usage: gemma3_native <LUJ-restored-model.safetensors> <token-id-csv> <next-tokens:0..64> [--expected-sha SHA256] [--logits-out new-file]");
 std::string expected="700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab";
 std::string logits_out;
 for(int i=4;i<argc;i++){std::string flag=argv[i];ck(i+1<argc,"flag requires an argument");
  if(flag=="--expected-sha")expected=argv[++i];
  else if(flag=="--logits-out")logits_out=argv[++i];
  else ck(false,"unknown option: "+flag);
 }
 const Model m(argv[1],expected);Gemma3 runtime(m);auto ids=parse_ids(argv[2]);int steps=std::stoi(argv[3]);ck(steps>=0&&steps<=64,"generation steps must be 0..64");
 std::cout<<"{\"model\":\"gemma3-270m-bf16\",\"prompt_token_ids\":[";
 for(size_t i=0;i<ids.size();i++){std::cout<<(i?",":"")<<ids[i];}
 std::cout<<"],\"generated_ids\":[";
 F hidden;for(size_t i=0;i<ids.size();i++){hidden=runtime.forward(ids[i]);if(i%10==0)std::cerr<<"TOKEN_PREFILL="<<i+1<<"/"<<ids.size()<<"\n";}
 F initial=runtime.logits(hidden);
 if(!logits_out.empty()){
  std::ifstream prior(logits_out,std::ios::binary);ck(!prior.good(),"logit output already exists");
  std::ofstream out(logits_out,std::ios::binary);ck(bool(out),"could not create logit output");
  out.write(reinterpret_cast<const char*>(initial.data()),std::streamsize(initial.size()*sizeof(float)));
  ck(bool(out),"logit output write failed");
 }
 for(int k=0;k<steps;k++){
  F scores=k==0?std::move(initial):runtime.logits(hidden);auto it=std::max_element(scores.begin(),scores.end());const uint32_t next=uint32_t(it-scores.begin());
  std::cout<<(k?",":"")<<next;std::cout.flush();
  if(next==1||next==106){break;}hidden=runtime.forward(next);
 }
 std::cout<<"],\"inference_source\":\"restored_LUJ1_weights\",\"comparison_to_transformers\":\"not_yet_measured\"}\n";
 }catch(const std::exception& ex){std::cerr<<"FAIL: "<<ex.what()<<"\n";return 1;}return 0;}
