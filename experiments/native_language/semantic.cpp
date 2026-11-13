// Native, closed-grammar semantic compiler. No Python or neural dependency.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
using I=std::int64_t;
struct Step { std::string op; I operand; bool reverse=false; };
I integer(const std::string& s) {
 I v; auto r=std::from_chars(s.data(),s.data()+s.size(),v);
 if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size()) throw std::runtime_error("invalid integer");
 return v;
}
std::string trim(std::string s) {
 auto a=s.find_first_not_of(" \t\r\n");
 if(a==std::string::npos)return "";
 return s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
Step parse(const std::string& s) {
 // Anchored productions consume the ENTIRE clause; no keyword guessing.
 struct Rule {const char* pattern; const char* op; bool reverse;};
 static const Rule rules[]={
 {R"(add (-?[0-9]+))","add",false},
 {R"(subtract (-?[0-9]+))","subtract",false},
 {R"(subtract the total from (-?[0-9]+))","subtract",true},
 {R"(take (-?[0-9]+) away from the total)","subtract",false},
 {R"(multiply by (-?[0-9]+))","multiply",false},
 {R"(divide (?:the running total )?by (-?[0-9]+))","divide",false},
 {R"(divide (-?[0-9]+) by the total)","divide",true},
 {R"(take the lower of this amount and (-?[0-9]+))","minimum",false},
 {R"(take the higher of this amount and (-?[0-9]+))","maximum",false},
 {R"(minimum with (-?[0-9]+))","minimum",false},
 {R"(maximum with (-?[0-9]+))","maximum",false},
 {R"(raise to power (-?[0-9]+))","power",false},
 {R"(remainder modulo (-?[0-9]+))","remainder",false}};
 std::smatch m; std::vector<Step> candidates;
 for(auto& r:rules) if(std::regex_match(s,m,std::regex(r.pattern)))
  candidates.push_back({r.op,integer(m[1]),r.reverse});
 if(candidates.size()!=1)throw std::runtime_error("unsupported or ambiguous clause: "+s);
 return candidates[0];
}
I checked(__int128 v) {
 if(v<std::numeric_limits<I>::min()||v>std::numeric_limits<I>::max())
  throw std::runtime_error("integer overflow");
 return static_cast<I>(v);
}
I execute(I state,const Step& s) {
 I a=s.reverse?s.operand:state,b=s.reverse?state:s.operand;
 if(s.op=="add")return checked((__int128)a+b);
 if(s.op=="subtract")return checked((__int128)a-b);
 if(s.op=="multiply")return checked((__int128)a*b);
 if(s.op=="minimum")return std::min(a,b);
 if(s.op=="maximum")return std::max(a,b);
 if(s.op=="divide"||s.op=="remainder") {
  if(!b)throw std::runtime_error("zero divisor");
  __int128 x=a,y=b,r=x%y;
  if(s.op=="remainder")return checked(r && ((r<0)!=(y<0))?r+y:r);
  if(r)throw std::runtime_error("inexact integer division");
  return checked(x/y);
 }
 if(s.op=="power") {
  if(b<0||b>63)throw std::runtime_error("exponent outside 0..63");
  I v=1; for(I i=0;i<b;++i)v=checked((__int128)v*a);return v;
 }
 throw std::runtime_error("unknown operation");
}
struct Result {I value;std::vector<Step> steps;};
Result solve(std::string q) {
 if(q.empty()||q.size()>2000)throw std::runtime_error("input size");
 for(char& c:q) {if(static_cast<unsigned char>(c)>127)throw std::runtime_error("ASCII grammar only"); if(c>='A'&&c<='Z')c+=32;}
 std::vector<std::string> clauses;
 size_t start=0;
 for(size_t p=0;p<=q.size();++p)if(p==q.size()||q[p]==';'){clauses.push_back(trim(q.substr(start,p-start)));start=p+1;}
 if(clauses.size()<2||clauses.size()>6)throw std::runtime_error("require 1..5 steps");
 std::smatch m;
 if(!std::regex_match(clauses[0],m,std::regex(R"(start with (-?[0-9]+))")))throw std::runtime_error("missing initial state");
 Result r{integer(m[1]),{}};
 // Complete semantic parsing before executing any instruction.
 for(size_t i=1;i<clauses.size();++i)r.steps.push_back(parse(clauses[i]));
 for(auto& s:r.steps)r.value=execute(r.value,s);
 return r;
}
#ifndef SEMANTIC_TEST
int main(int argc,char** argv) {
 if(argc!=2){std::cerr<<"usage: semantic 'Start with 12; multiply by 8; subtract 27; add 15'\n";return 2;}
 try {auto r=solve(argv[1]);std::cout<<"Within the declared arithmetic grammar, the result is "<<r.value<<".\n";
 for(auto&s:r.steps)std::cout<<s.op<<" "<<s.operand<<(s.reverse?" reversed":"")<<"\n";
 }catch(const std::exception&e){std::cerr<<"REFUSED: "<<e.what()<<"\n";return 1;}
}
#endif
