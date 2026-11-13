// Copyright 2026. LUJ1 Gemma 3 270M BF16/EXACT32 bridge.
// Native C++17, streaming, no Python at conversion or restoration time.
// Preserves the original model.safetensors BF16 bits and the LUJ1 legacy envelope.
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <zstd.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using U8 = std::uint8_t;
using B = std::vector<U8>;
using Hash = std::array<U8,32>;
static constexpr size_t RECORD_SIZE=20, STREAM_SIZE=108, INS_SIZE=22, MAX_RECORDS=1024;
static constexpr uint64_t EXPECTED_WEIGHTS=268098176;
static void check(bool condition, const std::string& msg){if(!condition)throw std::runtime_error(msg);}
static uint16_t u16(const U8* d){return uint16_t(d[0])|uint16_t(d[1])<<8;}
static uint32_t u32(const U8* d){return uint32_t(d[0])|uint32_t(d[1])<<8|uint32_t(d[2])<<16|uint32_t(d[3])<<24;}
static uint64_t u64(const U8* d){return uint64_t(u32(d))|(uint64_t(u32(d+4))<<32);}
static void add16(B& b,uint16_t x){b.push_back(U8(x));b.push_back(U8(x>>8));}
static void add32(B& b,uint32_t x){for(unsigned i=0;i<4;i++)b.push_back(U8(x>>(8*i)));}
static void add64(B& b,uint64_t x){for(unsigned i=0;i<8;i++)b.push_back(U8(x>>(8*i)));}
static void write_all(std::ostream& f,const B& b){f.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));check(bool(f),"write failure");}
static void read_exact(std::istream& f,U8* d,size_t n){f.read(reinterpret_cast<char*>(d),std::streamsize(n));check(f.gcount()==std::streamsize(n),"truncated input");}
static B read_file(const fs::path& p){std::ifstream f(p,std::ios::binary);check(bool(f),"missing file: "+p.string());return B(std::istreambuf_iterator<char>(f),{});}
static Hash digest(const B& b){Hash h{};SHA256(b.data(),b.size(),h.data());return h;}
static Hash digest(const U8* p,size_t n){Hash h{};SHA256(p,n,h.data());return h;}
static std::string hex(const U8* p,size_t n){static constexpr char H[]="0123456789abcdef";std::string s;for(size_t i=0;i<n;i++){s+=H[p[i]>>4];s+=H[p[i]&15];}return s;}
static std::string hex(const Hash& h){return hex(h.data(),h.size());}
static Hash from_hex(const std::string& s){check(s.size()==64,"expected a SHA256 digest");Hash out{};for(size_t i=0;i<32;i++){const auto cv=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;return -1;};int x=cv(s[2*i]),y=cv(s[2*i+1]);check(x>=0&&y>=0,"bad SHA256 hex");out[i]=U8(x*16+y);}return out;}
struct ShaStream{
 EVP_MD_CTX* ctx=EVP_MD_CTX_new();
 ShaStream(){check(ctx&&EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)==1,"SHA256 init failed");}
 ~ShaStream(){EVP_MD_CTX_free(ctx);}
 void add(const U8* ptr,size_t n){check(EVP_DigestUpdate(ctx,ptr,n)==1,"SHA256 update failed");}
 void add(const B& b){add(b.data(),b.size());}
 Hash finish(){Hash h{};unsigned n=0;check(EVP_DigestFinal_ex(ctx,h.data(),&n)==1&&n==32,"SHA256 final failed");return h;}
};
static void write_new(const fs::path& p,const B& v){check(!fs::exists(p),"will not overwrite "+p.string());std::ofstream f(p,std::ios::binary);check(bool(f),"cannot create "+p.string());write_all(f,v);}
static void write_new_text(const fs::path& p,const std::string& x){check(!fs::exists(p),"will not overwrite "+p.string());std::ofstream f(p);check(bool(f),"cannot create "+p.string());f<<x;check(bool(f),"failed writing text");}
static B pack20(uint64_t glyph_idx){B b(45,0);for(size_t t=0;t<20;t++){
 // Valid tokenizer vocabulary IDs, deliberately metadata rather than user input.
 const uint32_t id=uint32_t((glyph_idx*20+t)%262144ull);
 const size_t bit=t*18;for(size_t k=0;k<18;k++){if(id&(1u<<k))b[(bit+k)/8]|=U8(1u<<((bit+k)%8));}
 }return b;}
static void verify20(const U8* bytes,uint64_t glyph_idx){for(size_t t=0;t<20;t++){
 const size_t bit=t*18;uint32_t id=0;
 for(size_t k=0;k<18;k++){if(bytes[(bit+k)/8]&(1u<<((bit+k)%8)))id|=1u<<k;}
 check(id==uint32_t((glyph_idx*20+t)%262144ull),"token-ID lane mismatch");
 }}
static void append_instruction(B& body,uint64_t start,uint64_t valid){
 // Existing LUJ1 integer opcode ADD, two immediate operands: (tensor flat offset, valid weight count).
 // These opcodes are metadata descriptors, not neural reasoning instructions.
 body.push_back(1);body.push_back(0);add16(body,0xffff);add16(body,0xffff);add64(body,start);add64(body,valid);
}
static B pack_exact32_bf16(const U8* bf16,size_t valid){
 check(valid>0&&valid<=256,"invalid BF16 count");
 B planes(1024,0);
 for(size_t i=0;i<valid;i++){planes[2*256+i]=bf16[2*i];planes[3*256+i]=bf16[2*i+1];}
 uLongf bound=compressBound(uLong(planes.size()));B compressed(bound);
 check(compress2(compressed.data(),&bound,planes.data(),uLong(planes.size()),Z_BEST_SPEED)==Z_OK,"zlib encode failure");
 compressed.resize(bound);B payload(36,0);
 if(compressed.size()<planes.size()){payload.push_back(1);payload.insert(payload.end(),compressed.begin(),compressed.end());}
 else {payload.push_back(0);payload.insert(payload.end(),planes.begin(),planes.end());}
 return payload;
}
static B exact32_to_bf16(const U8* payload,size_t n,size_t valid,B* fp32){
 check(n>=37&&n<=1061,"invalid exact32 length");check(valid>0&&valid<=256,"invalid weight count");
 for(size_t i=0;i<36;i++)check(payload[i]==0,"unsupported nonzero predictor");
 B planes(1024);if(payload[36]==0){check(n==1061,"raw exact32 repair wrong length");std::copy(payload+37,payload+n,planes.begin());}
 else if(payload[36]==1){uLongf decoded=1024;check(uncompress(planes.data(),&decoded,payload+37,uLong(n-37))==Z_OK&&decoded==1024,"corrupt exact32 zlib repair");}
 else throw std::runtime_error("unknown exact32 repair mode");
 B bf16;bf16.reserve(valid*2);
 for(size_t i=0;i<256;i++){
 check(planes[i]==0&&planes[256+i]==0,"decoded FP32 was not an exact BF16 value");
 U8 a=planes[512+i],b=planes[768+i];
 uint32_t bits=uint32_t(a)<<16|uint32_t(b)<<24;
 float f;std::memcpy(&f,&bits,4);check(std::isfinite(f),"nonfinite BF16 weights forbidden");
 if(i>=valid)check(!a&&!b,"nonzero last-glyph padding");
 if(i<valid){bf16.push_back(a);bf16.push_back(b);}
 fp32->push_back(0);fp32->push_back(0);fp32->push_back(a);fp32->push_back(b);
 }
 return bf16;
}
static void complete_shard(const fs::path& dir,uint32_t shard_id,uint32_t record_count,const Hash& tok_sha,const B& body,const B& fp32){
 check(record_count>0&&record_count<=MAX_RECORDS,"bad shard size");
 B h;h.reserve(STREAM_SIZE+body.size());h.insert(h.end(),{'L','U','J','1'});h.push_back(1);h.push_back(18);add16(h,0);add32(h,record_count);
 h.insert(h.end(),tok_sha.begin(),tok_sha.end());const Hash source=digest(fp32),bh=digest(body);
 h.insert(h.end(),source.begin(),source.end());h.insert(h.end(),bh.begin(),bh.end());check(h.size()==STREAM_SIZE,"incorrect stream header");
 h.insert(h.end(),body.begin(),body.end());std::ostringstream ss;ss<<"part-"<<std::setw(6)<<std::setfill('0')<<shard_id<<".luj";
 write_new(dir/ss.str(),h);
}
static B pack_meta(uint64_t glyph_idx,unsigned count){
 check(count==20||count==21,"LUJ1 metadata requires 20 or 21 token IDs");
 B b((18*count+7)/8,0);
 for(size_t t=0;t<count;t++){
  const uint32_t id=uint32_t((glyph_idx*20+t)%262144ull);
  const size_t bit=t*18;
  for(size_t k=0;k<18;k++)if(id&(1u<<k))b[(bit+k)/8]|=U8(1u<<((bit+k)%8));
 }
 return b;
}
static void verify_meta(const U8* bytes,uint64_t glyph_idx,unsigned count){
 check(count==20||count==21,"invalid metadata token count");
 for(size_t t=0;t<count;t++){
  const size_t bit=t*18;uint32_t id=0;
  for(size_t k=0;k<18;k++)if(bytes[(bit+k)/8]&(1u<<((bit+k)%8)))id|=1u<<k;
  check(id==uint32_t((glyph_idx*20+t)%262144ull),"token-ID lane mismatch");
 }
 if(count==21)check((bytes[47]&0xfc)==0,"nonzero 21-token padding");
}
static float from_bf16(uint16_t b){uint32_t u=uint32_t(b)<<16;float out;std::memcpy(&out,&u,4);return out;}
static uint16_t to_bf16_rne(float x){
 check(std::isfinite(x),"nonfinite quantized model value");
 uint32_t u;std::memcpy(&u,&x,4);
 u+=0x7fff+((u>>16)&1u);
 return uint16_t(u>>16);
}
struct QuantStats{long double sqref=0,sqdiff=0;uint64_t exactly=0,count=0;float maxdiff=0;};
static B quantize_qbits_32(const B& raw,size_t valid,QuantStats& stats,int bits){
 check(bits==8||bits==10||bits==12||bits==14,"unsupported group32 bit width");
 const int qmax=(1<<(bits-1))-1;
 check(raw.size()==valid*2&&valid>0&&valid<=256,"bad Q8 group input");
 B output(raw.size());
 for(size_t base=0;base<valid;base+=32){
  const size_t stop=std::min(base+32,valid);
  float maxabs=0;
  for(size_t i=base;i<stop;i++){
   const float f=from_bf16(u16(raw.data()+2*i));
   check(std::isfinite(f),"source BF16 nonfinite");
   maxabs=std::max(maxabs,std::fabs(f));
  }
  const float scale=maxabs>0?maxabs/float(qmax):1.0f;
  for(size_t i=base;i<stop;i++){
   const float src=from_bf16(u16(raw.data()+2*i));
   const int q=std::clamp(int(std::nearbyint(src/scale)),-qmax,qmax);
   const uint16_t bits=to_bf16_rne(float(q)*scale);
   const float decoded=from_bf16(bits),e=decoded-src;
   output[2*i]=U8(bits);output[2*i+1]=U8(bits>>8);
   stats.sqref+=static_cast<long double>(src)*src;
   stats.sqdiff+=static_cast<long double>(e)*e;
   stats.exactly+=(bits==u16(raw.data()+2*i));
   stats.count++;
   stats.maxdiff=std::max(stats.maxdiff,std::fabs(e));
  }
 }
 return output;
}
static B quantize_q8_32(const B& raw,size_t valid,QuantStats& stats){return quantize_qbits_32(raw,valid,stats,8);}
static std::string trim(std::string s){while(!s.empty()&&(s.back()=='\n'||s.back()=='\r'||s.back()==' '))s.pop_back();return s;}
static void quantize_model(const fs::path& gbf,const fs::path& tokenizer,const fs::path& outdir,bool resume=false,bool preserve_embedding=false,int bits=8){
 check(bits==8||bits==10||bits==12||bits==14,"invalid quantization bits");
 if(!resume){check(!fs::exists(outdir),"refuse to overwrite Q8 output directory");fs::create_directories(outdir);}
 else check(fs::is_directory(outdir),"resume Q8 output directory missing");
 std::ifstream f(gbf,std::ios::binary);check(bool(f),"GBF source not found");
 U8 hdr[52];read_exact(f,hdr,52);check(std::memcmp(hdr,"GBF1",4)==0,"expected GBF1");
 const uint64_t modelsize=u64(hdr+4),headerlen=u64(hdr+12);
 check(headerlen>=8&&headerlen<1000000&&modelsize-headerlen==EXPECTED_WEIGHTS*2,"wrong Gemma3 GBF geometry");
 Hash sourcehash{};std::copy(hdr+20,hdr+52,sourcehash.begin());
 B header(headerlen);read_exact(f,header.data(),header.size());
 check(u64(header.data())==headerlen-8,"invalid source safetensors header");
 ShaStream source,quant;source.add(header);quant.add(header);
 const Hash toksha=digest(read_file(tokenizer));
 if(!resume){write_new(outdir/"safetensors.header",header);
 write_new_text(outdir/"tokenizer.sha256",hex(toksha)+"\n");
 write_new_text(outdir/"source_model.sha256",hex(sourcehash)+"\n");}
 else {check(read_file(outdir/"safetensors.header")==header,"Q8 resume header changed");
 B previous_tok=read_file(outdir/"tokenizer.sha256");check(trim(std::string(previous_tok.begin(),previous_tok.end()))==hex(toksha),"Q8 tokenizer changed");}
 QuantStats stats;uint32_t shard=0,records=0;uint64_t glyphid=0,total_weights=0,total_bytes=header.size();
 B body,fp32;body.reserve(850000);fp32.reserve(1024*1024);
 const uint64_t allglyphs=(EXPECTED_WEIGHTS+255)/256;
 uint64_t resume_glyphs=0;
 if(resume){
  for(uint32_t part=0;;part++){
   std::ostringstream path;path<<"part-"<<std::setw(6)<<std::setfill('0')<<part<<".luj";
   const auto file=outdir/path.str();if(!fs::exists(file))break;
   B block=read_file(file);check(block.size()>STREAM_SIZE&&u32(block.data()+8)==MAX_RECORDS,"cannot resume incomplete Q8 shard");
   Hash got=digest(block.data()+STREAM_SIZE,block.size()-STREAM_SIZE);
   check(std::equal(got.begin(),got.end(),block.data()+76),"corrupt Q8 resume shard");
   total_bytes+=block.size();resume_glyphs+=MAX_RECORDS;shard++;
  }
  check(resume_glyphs<allglyphs,"all Q8 glyphs already encoded");
  std::cerr<<"Q8_RESUME_PARTS="<<shard<<" SKIP_GLYPHS="<<resume_glyphs<<"\n";
 }
 for(;glyphid<allglyphs;glyphid++){
  U8 lengths[4];read_exact(f,lengths,4);const size_t valid=(glyphid+1==allglyphs)?128:256;
  const uint16_t n=u16(lengths),z=u16(lengths+2);
  check(n==2*valid&&z>0,"corrupt GBF frame length");
  B compressed(z),raw(n);read_exact(f,compressed.data(),compressed.size());
  check(ZSTD_decompress(raw.data(),raw.size(),compressed.data(),compressed.size())==n,"bad zstd weight frame");
  B original(n);for(size_t i=0;i<valid;i++){original[2*i]=raw[i];original[2*i+1]=raw[valid+i];}
  source.add(original);
  B q8;
  if(preserve_embedding && glyphid*512ull<335544320ull){
   q8=original;
   for(size_t i=0;i<valid;i++){float src=from_bf16(u16(original.data()+2*i));
    stats.sqref+=static_cast<long double>(src)*src;stats.exactly++;stats.count++;}
  }else q8=quantize_qbits_32(original,valid,stats,bits);
  quant.add(q8);
  if(glyphid<resume_glyphs){total_weights+=valid;continue;}
  B weightlane=pack_exact32_bf16(q8.data(),valid),checkfp;
  check(exact32_to_bf16(weightlane.data(),weightlane.size(),valid,&checkfp)==q8,"Q8 glyph decode mismatch");
  fp32.insert(fp32.end(),checkfp.begin(),checkfp.end());
  const unsigned token_count=(glyphid%5>=3)?21:20;
  B tokenlane=pack_meta(glyphid,token_count);
  add32(body,records);body.push_back(4);body.push_back(U8(token_count));add16(body,0);
  add32(body,uint32_t(weightlane.size()));add32(body,uint32_t(tokenlane.size()));add32(body,22);
  body.insert(body.end(),weightlane.begin(),weightlane.end());
  body.insert(body.end(),tokenlane.begin(),tokenlane.end());
  append_instruction(body,glyphid*256,valid);
  total_weights+=valid;records++;
  if(records==MAX_RECORDS||glyphid+1==allglyphs){
   const size_t bytes=STREAM_SIZE+body.size();complete_shard(outdir,shard++,records,toksha,body,fp32);
   total_bytes+=bytes;body.clear();fp32.clear();records=0;
  }
  if((glyphid+1)%131072==0)std::cerr<<"Q8_PROGRESS glyphs="<<glyphid+1<<"/"<<allglyphs<<"\n";
 }
 check(total_weights==EXPECTED_WEIGHTS&&f.peek()==EOF,"incomplete quantized model");
 const Hash sourceactual=source.finish(),quantactual=quant.finish();
 check(sourceactual==sourcehash,"GBF source hash mismatch; refusing Q8 model");
 write_new_text(outdir/"original.sha256",hex(quantactual)+"\n");
 const long double relative=std::sqrt(stats.sqdiff/stats.sqref);
 std::ostringstream proof;
 proof<<std::setprecision(16)
  <<"Q8_GLYPH_ENCODING=PASS\nSOURCE_CHECKPOINT_SHA256="<<hex(sourceactual)
  <<"\nQUANTIZED_CHECKPOINT_SHA256="<<hex(quantactual)
  <<"\nSOURCE_WEIGHTS="<<EXPECTED_WEIGHTS<<"\nGLYPHS="<<glyphid<<"\nSHARDS="<<shard
  <<"\nMETA_TOKENS_EVERY_FIVE=102\nINSTRUCTIONS_PER_FIVE=5"
  <<"\nQUANTIZATION="<<(preserve_embedding?"MIXED_EMBED_BF16_PLUS_Q8_GROUP32":"SIGNED_INT_PER_GROUP32_SCALE_FP32_RECONSTRUCT_BF16")
  <<"\nQUANTIZATION_BITS="<<bits
  <<"\nSTORAGE_CODEC=LUJ1_EXACT32_FOR_DEQUANTIZED_BF16_VALUES"
  <<"\nWEIGHT_RELATIVE_L2_ERROR="<<relative<<"\nWEIGHT_MAX_ABSOLUTE_ERROR="<<stats.maxdiff
  <<"\nBITWISE_PRESERVED_WEIGHTS="<<stats.exactly
  <<"\nSTORED_LUJ_BYTES="<<total_bytes<<"\nGPT_LEVEL_PROVEN=NO\n";
 write_new_text(outdir/"proof.txt",proof.str());std::cout<<proof.str();
}
static void encode(const fs::path& gbf,const fs::path& tokenizer,const fs::path& outdir,bool partial,uint64_t maxglyphs,bool resume=false){
 if(!resume){check(!fs::exists(outdir),"refuse to overwrite output directory");fs::create_directories(outdir);}
 else check(fs::is_directory(outdir),"resume directory missing");
 std::ifstream f(gbf,std::ios::binary);check(bool(f),"GBF input missing");
 U8 hdr[52];read_exact(f,hdr,52);check(std::memcmp(hdr,"GBF1",4)==0,"not a GBF1 glyph bank");
 const uint64_t orig_len=u64(hdr+4),original_header_length=u64(hdr+12);
 check(original_header_length>=8&&original_header_length<1000000,"GBF header length invalid");
 check(orig_len>=original_header_length&&orig_len-original_header_length==EXPECTED_WEIGHTS*2,"wrong model shape/data length");
 Hash expected{};std::copy(hdr+20,hdr+52,expected.begin());
 B safetensors_header(original_header_length);read_exact(f,safetensors_header.data(),safetensors_header.size());
 check(u64(safetensors_header.data())==original_header_length-8,"safetensors JSON header mismatch");
 check(!safetensors_header.empty()&&safetensors_header[8]=='{',"invalid tensor JSON");
 const Hash toksha=digest(read_file(tokenizer));
 if(!resume){write_new(outdir/"safetensors.header",safetensors_header);
 write_new_text(outdir/"original.sha256",hex(expected)+"\n");write_new_text(outdir/"tokenizer.sha256",hex(toksha)+"\n");}
 else {
  check(read_file(outdir/"safetensors.header")==safetensors_header,"resume header does not match source");
  B stored_orig=read_file(outdir/"original.sha256");check(trim(std::string(stored_orig.begin(),stored_orig.end()))==hex(expected),"resume checkpoint SHA mismatch");
  B stored_tok=read_file(outdir/"tokenizer.sha256");check(trim(std::string(stored_tok.begin(),stored_tok.end()))==hex(toksha),"resume tokenizer SHA mismatch");
 }
 ShaStream original;original.add(safetensors_header);
 uint64_t glyphid=0,total_weights=0;uint32_t shard=0,records=0;B body,fp32;body.reserve(800000);fp32.reserve(1048576);
 const uint64_t allglyphs=(EXPECTED_WEIGHTS+255)/256;
 const uint64_t target=(partial?std::min(maxglyphs,allglyphs):allglyphs);
 check(target>=1,"no glyphs requested");
 uint64_t resume_glyphs=0;
 if(resume){
   for(uint32_t part=0;;part++){
    std::ostringstream ss;ss<<"part-"<<std::setw(6)<<std::setfill('0')<<part<<".luj";
    fs::path p=outdir/ss.str();if(!fs::exists(p))break;
    B prior=read_file(p);check(prior.size()>STREAM_SIZE&&std::memcmp(prior.data(),"LUJ1",4)==0&&u32(prior.data()+8)==MAX_RECORDS,"resume shard not complete");
    check(std::equal(toksha.begin(),toksha.end(),prior.data()+12),"resume shard tokenizer mismatch");
    Hash bh=digest(prior.data()+STREAM_SIZE,prior.size()-STREAM_SIZE);
    check(std::equal(bh.begin(),bh.end(),prior.data()+76),"resume shard corrupted");
    shard++;resume_glyphs+=MAX_RECORDS;
   }
   check(resume_glyphs<allglyphs,"already completed or corrupt checkpoint");
   std::cerr<<"RESUMING_EXISTING_SHARDS="<<shard<<" GLYPHS="<<resume_glyphs<<"\n";
 }
 for(;glyphid<target;glyphid++){
  U8 lens[4];read_exact(f,lens,4);const uint16_t bytes=u16(lens),packed=u16(lens+2);
  const size_t valid=(glyphid+1==allglyphs)?128:256;
  check(bytes==valid*2&&packed>0,"invalid GBF frame lengths");
  B frame(packed);read_exact(f,frame.data(),frame.size());B raw(bytes);
  const size_t got=ZSTD_decompress(raw.data(),raw.size(),frame.data(),frame.size());
  check(!ZSTD_isError(got)&&got==bytes,"GBF zstd decompression failed");
  // GBF1 stores the low-byte plane followed by the high-byte plane for BF16.
  // Losslessly interleave planes to recover the original safetensors byte order.
  B reinterleaved(bytes);
  for(size_t j=0;j<valid;j++){reinterleaved[2*j]=raw[j];reinterleaved[2*j+1]=raw[valid+j];}
  raw.swap(reinterleaved);original.add(raw);
  if(glyphid<resume_glyphs){total_weights+=valid;continue;}
  B weightlane=pack_exact32_bf16(raw.data(),valid);
  // Validate in-process exact32 decode and source equivalence before creating a shard.
  B decodedfp32;B round=exact32_to_bf16(weightlane.data(),weightlane.size(),valid,&decodedfp32);
  check(round==raw,"BF16 round trip failed");fp32.insert(fp32.end(),decodedfp32.begin(),decodedfp32.end());
  add32(body,records);body.push_back(4);body.push_back(20);add16(body,0);
  add32(body,uint32_t(weightlane.size()));add32(body,45);add32(body,uint32_t(INS_SIZE));
  body.insert(body.end(),weightlane.begin(),weightlane.end());
  B tokens=pack20(glyphid);body.insert(body.end(),tokens.begin(),tokens.end());
  append_instruction(body,glyphid*256,valid);total_weights+=valid;records++;
  if(records==MAX_RECORDS||glyphid+1==target){complete_shard(outdir,shard++,records,toksha,body,fp32);body.clear();fp32.clear();records=0;}
  if((glyphid+1)%131072==0)std::cerr<<"ENCODE_PROGRESS glyphs="<<glyphid+1<<"/"<<target<<"\n";
 }
 bool complete=(glyphid==allglyphs);Hash reconstructed=original.finish();
 if(complete){check(total_weights==EXPECTED_WEIGHTS,"unexpected total model weights");check(reconstructed==expected,"GBF reconstruction failed original SHA256");
  check(f.peek()==EOF,"unexpected trailing bytes in GBF");}
 else check(partial,"unexpected incomplete conversion");
 std::ostringstream evidence;
 evidence<<"LUJ_GEMMA270_STATUS="<<(complete?"PASS":"PARTIAL_TEST_ONLY")<<"\n"
 <<"ORIGINAL_SAFETENSORS_SHA256="<<hex(expected)<<"\n"
 <<"RECONSTRUCTED_SHA256="<<(complete?hex(reconstructed):"NOT_APPLICABLE")<<"\n"
 <<"TOKENIZER_SHA256="<<hex(toksha)<<"\n"
 <<"GLYPHS="<<glyphid<<"\nWEIGHTS="<<total_weights<<"\nSHARDS="<<shard<<"\n"
 <<"GLYPH_WEIGHT_CODEC=EXACT32_BF16_BITS\nTOKENS_PER_GLYPH=20\n"
 <<"PROGRAM=ADD_IMMEDIATE_OFFSET_AND_VALID_COUNT\n"
 <<"TOKEN_ROLE=VOCABULARY_ID_METADATA_NOT_LANGUAGE_PROMPT\n"
 <<"GENERAL_LANGUAGE_UNDERSTANDING_PROVEN=NO\nGPT_LEVEL_INTELLIGENCE_PROVEN=NO\n";
 write_new_text(outdir/"proof.txt",evidence.str());std::cout<<evidence.str();
}
static void decode(const fs::path& directory,const fs::path& outfile,bool hashonly){
 B header=read_file(directory/"safetensors.header");check(header.size()>=8&&u64(header.data())==header.size()-8,"invalid safetensors.header");
 std::ifstream shafile(directory/"original.sha256"),tfile(directory/"tokenizer.sha256");check(shafile&&tfile,"missing sha files");std::string shastr,tokstr;std::getline(shafile,shastr);std::getline(tfile,tokstr);
 const Hash expected=from_hex(trim(shastr)),toksha=from_hex(trim(tokstr));
 check(!fs::exists(outfile)||hashonly,"refuse to overwrite reconstructed model");
 std::ofstream output;if(!hashonly){output.open(outfile,std::ios::binary);check(bool(output),"cannot open output file");write_all(output,header);}
 ShaStream sha;sha.add(header);uint64_t total_weights=0,glyphs=0;uint32_t shards=0;
 for(;;shards++){
  std::ostringstream name;name<<"part-"<<std::setw(6)<<std::setfill('0')<<shards<<".luj";
  fs::path path=directory/name.str();if(!fs::exists(path))break;
  B stream=read_file(path);check(stream.size()>=STREAM_SIZE,"short LUJ1 shard");const U8* h=stream.data();
  check(std::memcmp(h,"LUJ1",4)==0&&h[4]==1&&h[5]==18&&u16(h+6)==0,"incompatible LUJ1 header");
  uint32_t count=u32(h+8);check(count>=1&&count<=MAX_RECORDS,"bad shard count");
  check(std::equal(toksha.begin(),toksha.end(),h+12),"tokenizer digest mismatched");
  const auto body_hash=digest(h+STREAM_SIZE,stream.size()-STREAM_SIZE);
  check(std::equal(body_hash.begin(),body_hash.end(),h+76),"LUJ1 body integrity failure");
  ShaStream fpsha;size_t pos=STREAM_SIZE;
  for(uint32_t k=0;k<count;k++){
   check(pos+RECORD_SIZE<=stream.size(),"truncated record");
   const U8* rec=stream.data()+pos;
   uint32_t idx=u32(rec),codec=rec[4],nt=rec[5],res=u16(rec+6),nw=u32(rec+8),nb=u32(rec+12),ni=u32(rec+16);
   check(idx==k&&codec==4&&(nt==20||nt==21)&&res==0&&nb==(18*nt+7)/8&&ni==22&&nw>=37&&nw<=1061,"unsupported LUJ1 record");
   pos+=RECORD_SIZE;check(nw+nb+ni<=stream.size()-pos,"truncated glyph payload");
   const U8* w=stream.data()+pos,*t=w+nw,*ins=t+nb;
   verify_meta(t,glyphs,nt);
   check(ins[0]==1&&ins[1]==0&&u16(ins+2)==65535&&u16(ins+4)==65535,"unsupported glyph program descriptor");
   uint64_t offset=u64(ins+6),valid=u64(ins+14);
   check(offset==glyphs*256&&valid>=1&&valid<=256,"incorrect weight offset/count");
   check(valid==256||(valid==128&&total_weights+valid==EXPECTED_WEIGHTS),"invalid final glyph size");
   B fp32;fp32.reserve(1024);B bf16=exact32_to_bf16(w,nw,size_t(valid),&fp32);
   fpsha.add(fp32);sha.add(bf16);if(!hashonly)write_all(output,bf16);
   total_weights+=valid;glyphs++;pos+=nw+nb+ni;
  }
  check(pos==stream.size(),"trailing bytes in glyph shard");
  Hash got=fpsha.finish();check(std::equal(got.begin(),got.end(),h+44),"LUJ1 source weights SHA256 mismatch");
  if(shards%128==127)std::cerr<<"DECODE_PROGRESS shards="<<shards+1<<" glyphs="<<glyphs<<"\n";
 }
 check(shards>=1&&glyphs==(EXPECTED_WEIGHTS+255)/256&&total_weights==EXPECTED_WEIGHTS,"missing model weights/shards");
 Hash actual=sha.finish();check(expected==actual,"reconstructed model SHA256 differs from original");
 if(!hashonly){output.close();check(bool(output),"output close failed");}
 std::cout<<"LUJ_GEMMA270_RESTORE=PASS\nORIGINAL_MODEL_SHA256="<<hex(expected)<<"\nRESTORED_MODEL_SHA256="<<hex(actual)
 <<"\nGLYPHS="<<glyphs<<"\nWEIGHTS="<<total_weights<<"\nSHARDS="<<shards<<"\n";
}
// Upgrade a previously verified 20-token-per-glyph EXACT32 LUJ1 stream to
// 20/20/20/21/21 metadata tokens per five glyphs, retaining every weight bit.
static void upgrade102(const fs::path& input,const fs::path& output){
 check(fs::is_directory(input),"source glyph directory missing");
 check(!fs::exists(output),"output directory already exists");
 fs::create_directories(output);
 const B header=read_file(input/"safetensors.header");
 const B digest_orig=read_file(input/"original.sha256");
 const B digest_tok=read_file(input/"tokenizer.sha256");
 const Hash tok=from_hex(trim(std::string(digest_tok.begin(),digest_tok.end())));
 write_new(output/"safetensors.header",header);
 write_new(output/"original.sha256",digest_orig);
 write_new(output/"tokenizer.sha256",digest_tok);
 uint64_t total_glyphs=0,all_weight_count=0,token_count=0;
 unsigned shardid=0;
 for(;;shardid++){
  std::ostringstream name;name<<"part-"<<std::setw(6)<<std::setfill('0')<<shardid<<".luj";
  const auto path=input/name.str();if(!fs::exists(path))break;
  B old=read_file(path);check(old.size()>=STREAM_SIZE,"short old LUJ stream");
  check(std::memcmp(old.data(),"LUJ1",4)==0&&old[4]==1&&old[5]==18&&u16(old.data()+6)==0,"incompatible old LUJ header");
  const uint32_t count=u32(old.data()+8);check(count>=1&&count<=1024,"bad old glyph count");
  check(std::equal(tok.begin(),tok.end(),old.data()+12),"old tokenizer digest changed");
  const Hash bodyhash=digest(old.data()+STREAM_SIZE,old.size()-STREAM_SIZE);
  check(std::equal(bodyhash.begin(),bodyhash.end(),old.data()+76),"old body checksum failed");
  B body,fp32;body.reserve(old.size()+count*4);fp32.reserve(count*1024);
  size_t pos=STREAM_SIZE;
  for(uint32_t k=0;k<count;k++){
   check(pos+RECORD_SIZE<=old.size(),"truncated old glyph record");
   const U8* rec=old.data()+pos;
   const uint32_t idx=u32(rec),codec=rec[4],nt=rec[5],nw=u32(rec+8),nb=u32(rec+12),ni=u32(rec+16);
   check(idx==k&&codec==4&&nt==20&&u16(rec+6)==0&&nb==45&&ni==22&&nw>=37&&nw<=1061,"old source not EXACT32/20");
   pos+=RECORD_SIZE;check(nw+nb+ni<=old.size()-pos,"old glyph missing data");
   const U8* w=old.data()+pos,*token=w+nw,*instruction=token+nb;
   verify_meta(token,total_glyphs,20);
   const uint64_t offset=u64(instruction+6),n=u64(instruction+14);
   check(instruction[0]==1&&instruction[1]==0&&u16(instruction+2)==65535&&u16(instruction+4)==65535,"old glyph program invalid");
   check(offset==total_glyphs*256&&n>0&&n<=256,"old glyph program offsets invalid");
   B decoded;
   (void)exact32_to_bf16(w,nw,size_t(n),&decoded);
   fp32.insert(fp32.end(),decoded.begin(),decoded.end());
   const unsigned newer_count=total_glyphs%5>=3?21:20;
   const B updated_tokens=pack_meta(total_glyphs,newer_count);
   add32(body,k);body.push_back(4);body.push_back(U8(newer_count));add16(body,0);
   add32(body,nw);add32(body,uint32_t(updated_tokens.size()));add32(body,22);
   body.insert(body.end(),w,w+nw);
   body.insert(body.end(),updated_tokens.begin(),updated_tokens.end());
   body.insert(body.end(),instruction,instruction+22);
   total_glyphs++;all_weight_count+=n;token_count+=newer_count;
   pos+=nw+nb+ni;
  }
  check(pos==old.size(),"old LUJ trailing bytes");
  const Hash unpackhash=digest(fp32);check(std::equal(unpackhash.begin(),unpackhash.end(),old.data()+44),"old FP32 glyph tensor hash mismatch");
  complete_shard(output,shardid,count,tok,body,fp32);
  if((shardid+1)%128==0)std::cerr<<"UPGRADE102_SHARDS="<<shardid+1<<"\n";
 }
 check(total_glyphs==(EXPECTED_WEIGHTS+255)/256&&all_weight_count==EXPECTED_WEIGHTS,"missing glyph during upgrade");
 const uint64_t expected_tokens=102*(total_glyphs/5)+(total_glyphs%5==0?0:
  total_glyphs%5<=3?20*(total_glyphs%5):81);
 check(token_count==expected_tokens,"102-per-five metadata distribution failed");
 std::ostringstream proof;
 proof<<"EXACT_LUJ1_UPGRADE_102_TOKENS_PER_FIVE=PASS\n"
 <<"WEIGHTS="<<all_weight_count<<"\nGLYPHS="<<total_glyphs<<"\nTOKENS="<<token_count<<"\n"
 <<"TOKEN_PATTERN=20,20,20,21,21\nINSTRUCTIONS_PER_FIVE=5\n"
 <<"BF16_WEIGHT_PAYLOAD_CHANGE=NO\nMODEL_SHA256="<<trim(std::string(digest_orig.begin(),digest_orig.end()))<<"\n";
 write_new_text(output/"proof.txt",proof.str());std::cout<<proof.str();
}
static void selftest(){
  B raw(512);for(size_t i=0;i<256;i++){
    const uint16_t code=uint16_t(((i*13)%200)-100+0x3f80);
    raw[2*i]=U8(code);raw[2*i+1]=U8(code>>8);
  }
  B encoded=pack_exact32_bf16(raw.data(),256),fp32;
  B recovered=exact32_to_bf16(encoded.data(),encoded.size(),256,&fp32);
  check(recovered==raw&&fp32.size()==1024,"EXACT32 selftest failed");
  B token=pack20(10000);verify20(token.data(),10000);
  B corrupt=encoded;corrupt[0]=1;bool rejected=false;
  try{B ignored;exact32_to_bf16(corrupt.data(),corrupt.size(),256,&ignored);}catch(const std::exception&){rejected=true;}
  check(rejected,"altered codec predictor accepted");
  QuantStats stats;B quantized=quantize_q8_32(raw,256,stats);B quant_lane=pack_exact32_bf16(quantized.data(),256),quant_fp32;
  check(exact32_to_bf16(quant_lane.data(),quant_lane.size(),256,&quant_fp32)==quantized,"Q8 glyph decode failed");
  check(stats.count==256&&stats.sqref>0&&stats.sqdiff>=0,"Q8 statistics invalid");
 for(int bits:{10,12,14}){QuantStats ss;B higher=quantize_qbits_32(raw,256,ss,bits);B lane=pack_exact32_bf16(higher.data(),256),fp32tmp;check(exact32_to_bf16(lane.data(),lane.size(),256,&fp32tmp)==higher,"extended bit quant decode mismatch");check(ss.count==256,"extended bit stats invalid");}
  unsigned tokens=0;for(uint64_t i=0;i<5;i++){
   unsigned n=i%5>=3?21:20;B t=pack_meta(i,n);check(t.size()==(18*n+7)/8,"token lane size mismatch");
   verify_meta(t.data(),i,n);tokens+=n;
  }
  check(tokens==102,"five-glyph token count mismatch");
  std::cout<<"LUJ_EXACT32_BF16_SELFTEST=PASS\nLUJ_Q8_GROUP32_SELFTEST=PASS\nTOKEN_DISTRIBUTION_FIVE=20,20,20,21,21\nWEIGHTS_PER_GLYPH=256\n";
}
int main(int argc,char** argv){try{
 if(argc<2)throw std::runtime_error("usage: luj_gemma270 encode <model.gbf> <tokenizer.json> <new-dir> | decode <dir> <model.safetensors> | verify <dir> | sample <gbf> <tokenizer.json> <new-dir> <glyphcount>");
 std::string cmd=argv[1];
 if(cmd=="selftest"&&argc==2)selftest();
 else if(cmd=="encode"&&argc==5)encode(argv[2],argv[3],argv[4],false,0);
 else if(cmd=="quantize-q8"&&argc==5)quantize_model(argv[2],argv[3],argv[4]);
 else if(cmd=="resume-q8"&&argc==5)quantize_model(argv[2],argv[3],argv[4],true);
 else if(cmd=="quantize-q10"&&argc==5)quantize_model(argv[2],argv[3],argv[4],false,false,10);
 else if(cmd=="resume-q10"&&argc==5)quantize_model(argv[2],argv[3],argv[4],true,false,10);
 else if(cmd=="quantize-q12"&&argc==5)quantize_model(argv[2],argv[3],argv[4],false,false,12);
 else if(cmd=="resume-q12"&&argc==5)quantize_model(argv[2],argv[3],argv[4],true,false,12);
 else if(cmd=="quantize-q14"&&argc==5)quantize_model(argv[2],argv[3],argv[4],false,false,14);
 else if(cmd=="resume-q14"&&argc==5)quantize_model(argv[2],argv[3],argv[4],true,false,14);
 else if(cmd=="upgrade102"&&argc==4)upgrade102(argv[2],argv[3]);
 else if(cmd=="mixed-q8"&&argc==5)quantize_model(argv[2],argv[3],argv[4],false,true);
 else if(cmd=="resume-mixed-q8"&&argc==5)quantize_model(argv[2],argv[3],argv[4],true,true);
 else if(cmd=="sample"&&argc==6)encode(argv[2],argv[3],argv[4],true,std::stoull(argv[5]));
 else if(cmd=="resume"&&argc==5)encode(argv[2],argv[3],argv[4],false,0,true);
 else if(cmd=="decode"&&argc==4)decode(argv[2],argv[3],false);
 else if(cmd=="verify"&&argc==3)decode(argv[2],"",true);
 else throw std::runtime_error("invalid command arguments; run without arguments for usage");
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}return 0;}
