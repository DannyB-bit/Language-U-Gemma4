// Native LUJ1 QAT4 glyph planner: five 256-weight/20-21-token glyphs.
// C++17 + OpenSSL SHA-256. No Python or NumPy in the inference process.
// This is a bounded arithmetic-language planner, not a GPT-class transformer.
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using Byte = uint8_t;
using Bytes = std::vector<Byte>;
static void fail(const std::string& why) { throw std::runtime_error(why); }
static uint16_t u16(const Bytes& b, size_t p) { if(p+2>b.size()) fail("truncated u16");return uint16_t(b[p])|uint16_t(b[p+1])<<8; }
static uint32_t u32(const Bytes& b, size_t p) { if(p+4>b.size()) fail("truncated u32");return uint32_t(b[p])|uint32_t(b[p+1])<<8|uint32_t(b[p+2])<<16|uint32_t(b[p+3])<<24; }
static uint64_t u64(const Bytes& b,size_t p){return uint64_t(u32(b,p))|(uint64_t(u32(b,p+4))<<32);}
static int64_t i64(const Bytes& b,size_t p) { const uint64_t v=u64(b,p);int64_t out;std::memcpy(&out,&v,8);return out; }
static void put16(Bytes& b,size_t p,uint16_t v){b.at(p)=v&255;b.at(p+1)=(v>>8)&255;}
static void put64(Bytes& b,size_t p,uint64_t v){for(int i=0;i<8;++i)b.at(p+i)=Byte(v>>(8*i));}
static std::array<Byte,32> sha(const Byte* p,size_t n){std::array<Byte,32> h{};SHA256(p,n,h.data());return h;}
static std::array<Byte,32> sha(const Bytes& b){return sha(b.data(),b.size());}
static std::string hex(const Byte* b,size_t n){static const char* d="0123456789abcdef";std::string s;for(size_t i=0;i<n;++i){s+=d[b[i]>>4];s+=d[b[i]&15];}return s;}
static Bytes read_file(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f)fail("cannot open "+p);return Bytes(std::istreambuf_iterator<char>(f),{});}
static void write_new(const std::string& p,const Bytes& data){std::ifstream check(p,std::ios::binary);if(check.good())fail("output exists: "+p);std::ofstream f(p,std::ios::binary);if(!f)fail("cannot write "+p);f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));if(!f)fail("write failed");}

// RFC 7693 BLAKE2s, parameterized 8-byte digest to match Python hashlib.blake2s(digest_size=8).
static uint32_t rotr(uint32_t w,unsigned c){return (w>>c)|(w<<(32-c));}
static std::array<Byte,8> blake2s8(const std::string& s){
 static constexpr uint32_t IV[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 static constexpr Byte SIG[10][16]={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},{11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},{7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},{9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},{2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},{12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},{6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},{10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0}};
 uint32_t h[8];std::copy(IV,IV+8,h);h[0]^=0x01010008; // outlen=8, fanout=1, depth=1
 size_t pos=0;
 do {
  const size_t remaining=s.size()-pos,blocklen=std::min<size_t>(64,remaining);
  Byte buf[64]{}; if(blocklen)std::memcpy(buf,s.data()+pos,blocklen);
  uint32_t m[16]{};for(int i=0;i<16;++i)m[i]=uint32_t(buf[4*i])|uint32_t(buf[4*i+1])<<8|uint32_t(buf[4*i+2])<<16|uint32_t(buf[4*i+3])<<24;
  uint32_t v[16];std::copy(h,h+8,v);std::copy(IV,IV+8,v+8);uint64_t t=pos+blocklen;
  v[12]^=uint32_t(t);v[13]^=uint32_t(t>>32);if(pos+blocklen==s.size())v[14]=~v[14];
  auto G=[&](int a,int b,int c,int d,uint32_t x,uint32_t y){v[a]=v[a]+v[b]+x;v[d]=rotr(v[d]^v[a],16);v[c]+=v[d];v[b]=rotr(v[b]^v[c],12);v[a]=v[a]+v[b]+y;v[d]=rotr(v[d]^v[a],8);v[c]+=v[d];v[b]=rotr(v[b]^v[c],7);};
  for(int r=0;r<10;++r){const Byte* p=SIG[r];G(0,4,8,12,m[p[0]],m[p[1]]);G(1,5,9,13,m[p[2]],m[p[3]]);G(2,6,10,14,m[p[4]],m[p[5]]);G(3,7,11,15,m[p[6]],m[p[7]]);G(0,5,10,15,m[p[8]],m[p[9]]);G(1,6,11,12,m[p[10]],m[p[11]]);G(2,7,8,13,m[p[12]],m[p[13]]);G(3,4,9,14,m[p[14]],m[p[15]]);}
  for(int i=0;i<8;++i)h[i]^=v[i]^v[i+8];
  pos+=blocklen;
 }while(pos<s.size());
 std::array<Byte,8> out{};for(int i=0;i<8;++i)out[i]=Byte(h[i/4]>>(8*(i%4)));return out;
}

struct Glyph {std::array<float,256> weights{};std::vector<uint32_t> tokens;size_t instruction_offset=0;size_t record_end=0;};
struct Model {Bytes binary;std::array<Glyph,5> glyphs;std::array<float,1280> weights{};std::array<Byte,32> tokenizer_sha{};};
static Model load_model(const std::string& path){
 Model m;m.binary=read_file(path);const Bytes& b=m.binary;
 if(b.size()<108||std::memcmp(b.data(),"LUJ1",4)!=0||b[4]!=1||b[5]!=18||u16(b,6)!=0||u32(b,8)!=5)fail("invalid LUJ1 stream header");
 const std::string got=hex(sha(b).data(),32);
 if(got!="59841ffb1420237ea595a7f0f3645b407917170732b38fd554a4117825f3acac")fail("unrecognized model bank SHA256");
 std::copy(b.begin()+12,b.begin()+44,m.tokenizer_sha.begin());
 if(hex(m.tokenizer_sha.data(),32)!="cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f")fail("tokenizer binding mismatch");
 auto bodyhash=sha(b.data()+108,b.size()-108);
 if(!std::equal(bodyhash.begin(),bodyhash.end(),b.begin()+76))fail("LUJ1 body SHA256 mismatch");
 size_t pos=108;Bytes weight_raw;weight_raw.reserve(5120);
 for(size_t i=0;i<5;++i){
  if(pos+20>b.size())fail("truncated record");
  const uint32_t idx=u32(b,pos);int codec=b[pos+4],nt=b[pos+5];int reserved=u16(b,pos+6);uint32_t nw=u32(b,pos+8),nb=u32(b,pos+12),ni=u32(b,pos+16);pos+=20;
  if(idx!=i||codec!=1||nt<20||nt>21||reserved||nw!=144||nb!=uint32_t((18*nt+7)/8)||ni!=22||nw+nb+ni>b.size()-pos)fail("invalid QAT4 glyph record");
  auto& g=m.glyphs[i];const size_t wpos=pos;
  for(size_t j=0;j<256;++j){const Byte packed=b[wpos+j/2];const int code=int((j%2)?packed>>4:packed&15)-8;const uint32_t scbits=uint32_t(u16(b,wpos+128+(j/32)*2))<<16;
   float scale;std::memcpy(&scale,&scbits,4);if(!std::isfinite(scale)||scale<0)fail("bad BF16 QAT scale");float val=float(code)*scale;
   uint32_t bits;std::memcpy(&bits,&val,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;std::memcpy(&val,&bits,4);if(!std::isfinite(val))fail("nonfinite decoded weight");
   g.weights[j]=val;m.weights[i*256+j]=val;
   for(int k=0;k<4;++k)weight_raw.push_back(Byte(bits>>(8*k)));
  }
  pos+=nw;
  uint64_t acc=0;unsigned pending=0;size_t tpos=pos;for(int t=0;t<nt;++t){while(pending<18){if(tpos>=pos+nb)fail("token lane truncated");acc|=uint64_t(b[tpos++])<<pending;pending+=8;}g.tokens.push_back(uint32_t(acc&0x3ffff));acc>>=18;pending-=18;}
  if(acc||tpos!=pos+nb)fail("nonzero token lane padding");
  pos+=nb;
  g.instruction_offset=pos;
  // Program blueprint instructions must form [32,32,8,1,2].
  if(b[pos]!=1||b[pos+1]!=0||u16(b,pos+2)!=uint16_t(-1)||u16(b,pos+4)!=uint16_t(-1)||i64(b,pos+6)!=int64_t(std::array<int,5>{32,32,8,1,2}[i])||i64(b,pos+14)!=0)fail("wrong bank blueprint");
  pos+=22;g.record_end=pos;
 }
 if(pos!=b.size())fail("unexpected trailing data");
 auto recovered_sha=sha(weight_raw);
 if(!std::equal(recovered_sha.begin(),recovered_sha.end(),b.begin()+44))fail("reconstructed weights hash mismatch: "+hex(recovered_sha.data(),32));
 return m;
}

static std::string lower(std::string s){for(char& c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s;}
static std::vector<std::string> words(const std::string& s){std::vector<std::string> a;std::string w;for(char c:s){if(c>='a'&&c<='z')w+=c;else if(!w.empty()){a.push_back(w);w.clear();}}if(!w.empty())a.push_back(w);return a;}
static std::array<float,32> features(const std::string& clause){
 // Same numerical masking and signed-hash features as frozen PR #22.
 const std::unordered_set<std::string> stop={"a","an","the","this","that","value","total","current","running","amount","by","with","from","of","to","into","and","another","extra","additional","now","units","whole","number"};
 std::string s=lower(clause),masked;for(size_t i=0;i<s.size();){if(std::isdigit(static_cast<unsigned char>(s[i]))||(s[i]=='-'&&i+1<s.size()&&std::isdigit(static_cast<unsigned char>(s[i+1])))){if(s[i]=='-')++i;while(i<s.size()&&std::isdigit(static_cast<unsigned char>(s[i])))++i;masked+=" number ";}else masked+=s[i++];}
 std::vector<std::string> ws;for(auto& w:words(masked))if(!stop.count(w))ws.push_back(w);
 std::array<float,32> v{};
 auto add=[&](const std::string& atom,float weight){auto h=blake2s8(atom);uint32_t index=(uint32_t(h[0])|uint32_t(h[1])<<8|uint32_t(h[2])<<16|uint32_t(h[3])<<24)%31;v[index]+=((h[4]&1)?weight:-weight);};
 for(size_t i=0;i<ws.size();++i){add("w:"+ws[i],1.0f);for(size_t k=0;k+3<=ws[i].size();++k)add("c:"+ws[i].substr(k,3),0.3f);if(i+1<ws.size())add("b:"+ws[i]+" "+ws[i+1],0.4f);}
 float sum=0;for(float x:v)sum+=x*x;float norm=std::sqrt(sum);if(norm)for(float& x:v)x/=norm;v[31]=1;return v;
}
static const std::array<std::string,8> OP_NAMES={"add","subtract","multiply","divide","minimum","maximum","power","remainder"};
static std::array<float,8> predict(const Model& m,const std::string& s){
 auto x=features(s);std::array<float,32> hidden{};for(int h=0;h<32;++h){float z=0;for(int i=0;i<32;++i)z+=x[i]*m.weights[i*32+h];hidden[h]=std::tanh(z);}
 std::array<float,8> logits{};for(int j=0;j<8;++j){float z=0;for(int h=0;h<32;++h)z+=hidden[h]*m.weights[1024+h*8+j];logits[j]=z;}
 float mx=*std::max_element(logits.begin(),logits.end()),tot=0;for(float& z:logits){z=std::exp(z-mx);tot+=z;}for(float& z:logits)z/=tot;return logits;
}
// A deterministic semantic constraint only for unambiguous comparison phrasing.
// It is a safety check around the learned planner, not a claim of learned language comprehension.
static int comparison_guard(const std::string& clause){
 const auto w=words(lower(clause));auto has=[&](const std::string& term){return std::find(w.begin(),w.end(),term)!=w.end();};
 bool ismin=has("minimum")||has("smaller")||has("lesser")||has("lower")||has("cap");
 bool ismax=has("maximum")||has("larger")||has("greater")||has("higher")||has("floor");
 if(has("at")){for(size_t i=0;i+1<w.size();++i)if(w[i]=="at"&&w[i+1]=="most")ismin=true;else if(w[i]=="at"&&w[i+1]=="least")ismax=true;}
 const bool isremainder=has("remainder")||has("mod")||has("modulo")||has("modulus")||has("leftover");
 if(int(ismin)+int(ismax)+int(isremainder)>1)fail("conflicting operation semantics");
 return ismin?4:ismax?5:isremainder?7:-1;
}
static bool is_negation(const std::string& c){auto w=words(lower(c));for(auto& x:w)if(x=="not"||x=="never"||x=="unless"||x=="except"||x=="don"||x=="instead"||x=="ignore")return true;return false;}
static int64_t parse_number(const std::string& s){size_t used=0;long long n=0;try{n=std::stoll(s,&used,10);}catch(...){fail("invalid/overflow integer");}if(used!=s.size())fail("bad integer");return int64_t(n);}
struct Clause{std::string text;int64_t operand;};
struct Question{int64_t start;std::vector<Clause> clauses;};
static std::string trim(const std::string& s){const size_t a=s.find_first_not_of(" \t\n\r"),b=s.find_last_not_of(" \t\n\r");return a==std::string::npos?"":s.substr(a,b-a+1);}
static Question parse_question(const std::string& prompt){
 if(prompt.size()>2000||is_negation(prompt))fail("unsupported question / negation");
 std::string src=trim(prompt);if(!src.empty()&&src.back()=='.')src.pop_back();
 std::vector<std::string> seg;size_t i=0;while(true){size_t next=src.find(';',i);seg.push_back(trim(src.substr(i,next==std::string::npos?next:next-i)));if(next==std::string::npos)break;i=next+1;}
 if(seg.size()<2||seg.size()>6)fail("expected start and 1-5 clauses");
 std::smatch match;
 if(!std::regex_match(seg[0],match,std::regex("[Ss][Tt][Aa][Rr][Tt] +[Ww][Ii][Tt][Hh] +(-?[0-9]+)")))fail("invalid initial-value grammar");
 Question q{parse_number(match[1].str()),{}};
 for(size_t j=1;j<seg.size();++j){const std::string s=seg[j];for(char c:s)if(!std::isalnum(static_cast<unsigned char>(c))&&c!=' '&&c!='-'&&c!='\t')fail("unsupported clause characters");
  const std::regex number("-?[0-9]+");int found=0;std::string value;
  for(auto it=std::sregex_iterator(s.begin(),s.end(),number);it!=std::sregex_iterator();++it){const size_t p=it->position(),end=p+it->length();if((p&&std::isalnum(static_cast<unsigned char>(s[p-1])))||(end<s.size()&&std::isalnum(static_cast<unsigned char>(s[end]))))continue;value=it->str();++found;}
  if(found!=1)fail("clause needs exactly one isolated integer");
  q.clauses.push_back({s,parse_number(value)});
 }
 return q;
}
static int64_t safe_math(int op,int64_t x,int64_t y){
 constexpr __int128 lo=std::numeric_limits<int64_t>::min(),hi=std::numeric_limits<int64_t>::max();
 __int128 v=0;
 switch(op){case 0:v=(__int128)x+y;break;case 1:v=(__int128)x-y;break;case 2:v=(__int128)x*y;break;
 case 3:if(y==0||(x==std::numeric_limits<int64_t>::min()&&y==-1)||x%y)fail("inexact/zero/overflow division");v=(__int128)x/y;break;
 case 4:v=std::min(x,y);break;case 5:v=std::max(x,y);break;
 case 6:if(y<0||y>63)fail("invalid exponent");{__int128 cur=x;v=1;for(int64_t i=0;i<y;++i){v*=cur;if(v<lo||v>hi)fail("power overflows int64");}}break;
 case 7:if(y==0)fail("remainder by zero");{__int128 r=(__int128)x%y;if(r!=0&&((r<0)!=(y<0)))r+=y;v=r;}break;
 default:fail("unsupported operation");}
 if(v<lo||v>hi)fail("int64 overflow");
 return int64_t(v);
}
struct Answer{int64_t value;std::vector<int> ops;std::vector<float> confidence;Bytes output;int semantic_overrides=0;};
static Answer solve(const Model& m,const std::string& prompt){
 Question q=parse_question(prompt);Answer a{q.start,{}, {},m.binary};
 for(size_t i=0;i<q.clauses.size();++i){auto p=predict(m,q.clauses[i].text);auto it=std::max_element(p.begin(),p.end());int choice=int(it-p.begin());float certainty=*it;
  int semantic=comparison_guard(q.clauses[i].text);if(semantic>=0){choice=semantic;certainty=1.0f;++a.semantic_overrides;}
  if(certainty<0.55f)fail("low-confidence neural planner: abstained");
  a.value=safe_math(choice,a.value,q.clauses[i].operand);a.ops.push_back(choice);a.confidence.push_back(certainty);
 }
 // Rewrite only instruction lanes, preserve the original 256 weights and token IDs per glyph.
 for(size_t i=0;i<5;++i){const size_t p=m.glyphs[i].instruction_offset;int op=0;int64_t left=0,right=0;int16_t leftref=-1;
  if(i<a.ops.size()){const int mapping[]={1,2,3,6,7,8,9,10};op=mapping[a.ops[i]];leftref=i==0?-1:int16_t(i-1);left=i==0?q.start:0;right=q.clauses[i].operand;}else{op=1;leftref=i==0?-1:int16_t(i-1);right=0;}
  a.output[p]=Byte(op);a.output[p+1]=0;put16(a.output,p+2,uint16_t(leftref));put16(a.output,p+4,uint16_t(-1));put64(a.output,p+6,uint64_t(left));put64(a.output,p+14,uint64_t(right));
 }
 const auto digest=sha(a.output.data()+108,a.output.size()-108);std::copy(digest.begin(),digest.end(),a.output.begin()+76);
 return a;
}
static std::string json_escape(const std::string& s){std::string o;for(char c:s){if(c=='"'||c=='\\')o+='\\';if(c=='\n')o+="\\n";else o+=c;}return o;}
int main(int argc,char** argv){try{
 std::string path,question,out;bool inspect=false;
 for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--bank"&&i+1<argc)path=argv[++i];else if(arg=="--question"&&i+1<argc)question=argv[++i];else if(arg=="--chunk-out"&&i+1<argc)out=argv[++i];else if(arg=="--inspect")inspect=true;else fail("unknown or missing option: "+arg);}
 if(path.empty())fail("--bank is required");
 Model m=load_model(path);
 if(inspect){size_t n=0;for(auto& g:m.glyphs)n+=g.tokens.size();std::cout<<"{\"status\":\"verified\",\"glyphs\":5,\"weights\":1280,\"tokens\":"<<n<<",\"bytes\":"<<m.binary.size()<<",\"model_sha256\":\""<<hex(sha(m.binary).data(),32)<<"\"}\n";return 0;}
 if(question.empty())fail("--question is required");
 auto a=solve(m,question);if(!out.empty())write_new(out,a.output);
 std::cout<<"{\"status\":\"answered\",\"answer\":"<<a.value<<",\"operations\":[";for(size_t i=0;i<a.ops.size();++i){if(i)std::cout<<',';std::cout<<'"'<<OP_NAMES[a.ops[i]]<<'"';}std::cout<<"],\"confidence\":[";for(size_t i=0;i<a.confidence.size();++i){if(i)std::cout<<',';std::cout<<std::fixed<<std::setprecision(7)<<a.confidence[i];}std::cout<<"],\"neural_scores_computed\":true,\"neural_decisions_used\":"<<(a.ops.size()-a.semantic_overrides)<<",\"semantic_overrides\":"<<a.semantic_overrides<<",\"semantic_correctness_certified\":false,\"general_language_understanding_proven\":false,\"program_bytes\":"<<a.output.size()<<"}\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<"{\"status\":\"rejected\",\"error\":\""<<json_escape(e.what())<<"\"}\n";return 1;}}
