#define SEMANTIC_TEST
#include "semantic.cpp"
int main(){
 int n=0;
 auto expect=[&](const std::string&q,I v){if(solve(q).value!=v)throw std::runtime_error("wrong answer: "+q);++n;};
 auto reject=[&](const std::string&q){bool refused=false;try{solve(q);}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("accepted: "+q);++n;};
 expect("Start with 547; take the lower of this amount and 193; divide the running total by 1; take 449 away from the total",-256);
 expect("Start with 12; multiply by 8; subtract 27; add 15",84);
 expect("Start with 12; subtract the total from 27",15);
 expect("Start with 12; subtract 27",-15);
 expect("Start with 4; divide 20 by the total",5);
 expect("Start with -7; remainder modulo 3",2);
 expect("Start with 7; remainder modulo -3",-2);
 expect("Start with 2; raise to power 10",1024);
 expect("Start with 12; maximum with 27; minimum with 20",20);
 for(I a=-100;a<=100;++a)for(I b=-20;b<=20;++b){
  expect("Start with "+std::to_string(a)+"; add "+std::to_string(b),a+b);
  expect("Start with "+std::to_string(a)+"; subtract the total from "+std::to_string(b),b-a);
 }
 for(auto q:{"Start with 1; do not add 2","Start with 1; add 2 unless zero","Start with 1; add 2 subtract 3","Start with 1; add 2;","Start with 1; divide by 0","Start with 1; divide by 2","Start with 9223372036854775807; add 1","Start with -9223372036854775808; divide by -1","Start with 2; raise to power 63","Start with 1; add 2.5","Start with 1; add 999999999999999999999999"})reject(q);
 std::cout<<n<<" native semantic checks passed\n";
}
