// Cross-check the first-token full-vocabulary logits from two native model runs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
static std::vector<float> read(const char* path){
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("missing logits");
 f.seekg(0,std::ios::end);auto n=f.tellg();if(n!=262144*4)throw std::runtime_error("wrong Gemma3 vocabulary size");f.seekg(0);
 std::vector<float> x(262144);f.read(reinterpret_cast<char*>(x.data()),n);
 if(!f)throw std::runtime_error("truncated logits");
 return x;
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: compare_logits source.bin candidate.bin");
 auto a=read(argv[1]),b=read(argv[2]);
 double aa=0,bb=0,ab=0,err=0,mx=0,ma=-1e300,mb=-1e300;
 for(size_t i=0;i<a.size();i++){
  if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite logits");
  double x=a[i],y=b[i],d=y-x;aa+=x*x;bb+=y*y;ab+=x*y;err+=d*d;mx=std::max(mx,std::abs(d));
  ma=std::max(ma,x);mb=std::max(mb,y);
 }
 double za=0,zb=0,kl=0;
 for(size_t i=0;i<a.size();i++){za+=std::exp(double(a[i])-ma);zb+=std::exp(double(b[i])-mb);}
 for(size_t i=0;i<a.size();i++){
  double p=std::exp(double(a[i])-ma)/za;
  if(p>0)kl+=p*((a[i]-ma-std::log(za))-(b[i]-mb-std::log(zb)));
 }
 const auto argmax=[](const auto& a){return std::max_element(a.begin(),a.end())-a.begin();};
 std::cout.precision(12);
 std::cout<<"{\"source_argmax\":"<<argmax(a)<<",\"candidate_argmax\":"<<argmax(b)
 <<",\"argmax_match\":"<<(argmax(a)==argmax(b)?"true":"false")
 <<",\"cosine\":"<<(ab/std::sqrt(aa*bb))
 <<",\"relative_l2\":"<<std::sqrt(err/aa)
 <<",\"max_abs_error\":"<<mx<<",\"kl_source_candidate\":"<<kl<<"}\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
