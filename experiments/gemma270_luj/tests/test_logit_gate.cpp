// Synthetic negative control only; never substitutes for model output.
#include <cmath>
#include <fstream>
#include <vector>
int main(int argc,char** argv){
 if(argc!=3)return 1;
 std::vector<float> a(262144),b(262144);
 for(size_t i=0;i<a.size();++i)a[i]=b[i]=float(std::sin(double(i)*0.023));
 a[80]=b[80]=15.0f; b[80]=0.0f; b[81]=20.0f;
 std::ofstream(argv[1],std::ios::binary).write(reinterpret_cast<char*>(a.data()),a.size()*sizeof(float));
 std::ofstream(argv[2],std::ios::binary).write(reinterpret_cast<char*>(b.data()),b.size()*sizeof(float));
}