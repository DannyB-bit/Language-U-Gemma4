// Fail-closed logit parity gate for Gemma 3 270M. Comparison is only as independent as inputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t VOCAB = 262144;
static std::vector<float> read_logits(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("missing logits: ") + path);
    if (f.tellg() != std::streamoff(VOCAB * sizeof(float)))
        throw std::runtime_error("logit file size must be exactly 262144 FP32 values");
    f.seekg(0);
    std::vector<float> out(VOCAB);
    f.read(reinterpret_cast<char*>(out.data()), VOCAB * sizeof(float));
    if (!f) throw std::runtime_error("truncated logits");
    for (float v : out) if (!std::isfinite(v)) throw std::runtime_error("nonfinite logits");
    return out;
}
static double numeric(const char* arg) {
    std::string s(arg);
    size_t pos = 0;
    double value = std::stod(s, &pos);
    if (pos != s.size() || !std::isfinite(value))
        throw std::runtime_error("invalid finite threshold: " + s);
    return value;
}
int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error(
            "usage: compare_logits source.bin candidate.bin [--min-cosine X] "
            "[--max-relative-l2 X] [--max-abs-error X] [--max-kl X] [--require-argmax]");
        double min_cosine = -1, max_l2 = std::numeric_limits<double>::infinity(),
               max_abs = std::numeric_limits<double>::infinity(),
               max_kl = std::numeric_limits<double>::infinity();
        bool require_argmax = false;
        for (int i = 3; i < argc; ++i) {
            const std::string opt(argv[i]);
            if (opt == "--require-argmax") { require_argmax = true; continue; }
            if (i+1 >= argc) throw std::runtime_error("threshold missing for " + opt);
            const double v = numeric(argv[++i]);
            if (opt == "--min-cosine" && v >= -1 && v <= 1) min_cosine = v;
            else if (opt == "--max-relative-l2" && v >= 0) max_l2 = v;
            else if (opt == "--max-abs-error" && v >= 0) max_abs = v;
            else if (opt == "--max-kl" && v >= 0) max_kl = v;
            else throw std::runtime_error("unrecognized option or invalid range: " + opt);
        }
        const auto a = read_logits(argv[1]);
        const auto b = read_logits(argv[2]);
        double aa=0, bb=0, ab=0, err=0, mx=0;
        double ma = -std::numeric_limits<double>::infinity();
        double mb = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < VOCAB; ++i) {
            const double x=a[i], y=b[i], d=y-x;
            aa += x*x; bb += y*y; ab += x*y; err += d*d;
            mx = std::max(mx, std::abs(d));
            ma = std::max(ma, x); mb = std::max(mb, y);
        }
        if (aa == 0 || bb == 0) throw std::runtime_error("zero-norm logits are invalid for cosine");
        const double cosine = ab / std::sqrt(aa * bb);
        const double relative_l2 = std::sqrt(err / aa);
        double za=0, zb=0;
        for (size_t i=0; i<VOCAB; ++i) {
            za += std::exp(double(a[i])-ma);
            zb += std::exp(double(b[i])-mb);
        }
        double kl=0;
        for (size_t i=0; i<VOCAB; ++i) {
            const double p = std::exp(double(a[i])-ma)/za;
            if (p > 0) kl += p * ((double(a[i])-ma-std::log(za)) -
                                   (double(b[i])-mb-std::log(zb)));
        }
        kl = std::max(0.0, kl);
        auto argmax = [](const auto& v){return std::max_element(v.begin(),v.end()) - v.begin();};
        const auto ai=argmax(a),bi=argmax(b);
        const bool pass = cosine >= min_cosine && relative_l2 <= max_l2 &&
                          mx <= max_abs && kl <= max_kl &&
                          (!require_argmax || ai == bi);
        std::cout << std::setprecision(12)
            << "{\"source_argmax\":" << ai
            << ",\"candidate_argmax\":" << bi
            << ",\"argmax_match\":" << (ai==bi?"true":"false")
            << ",\"cosine\":" << cosine
            << ",\"relative_l2\":" << relative_l2
            << ",\"max_abs_error\":" << mx
            << ",\"kl_source_candidate\":" << kl
            << ",\"thresholds_enforced\":"
            << ((min_cosine>-1 || std::isfinite(max_l2) || std::isfinite(max_abs) ||
                  std::isfinite(max_kl) || require_argmax)?"true":"false")
            << ",\"gate_pass\":" << (pass?"true":"false") << "}\n";
        return pass ? 0 : 2;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
