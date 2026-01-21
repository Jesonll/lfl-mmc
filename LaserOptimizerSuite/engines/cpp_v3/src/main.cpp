/**
 * LASER SPEED PLANNER v3.2 (C++17)
 * 
 * FEATURES:
 * - Robust SVG Import (Curves + Safe Parsing)
 * - Coordinate Normalization (Y-Flip + Scaling)
 * - G-Code & HTML Export
 * - Physical Machine Time Estimation
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <sstream>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cctype>
#include <omp.h>
#include <Eigen/Dense>
#include <fmt/core.h>

// --- CORE TYPES ---
using Vec2 = Eigen::Vector2d;

struct Segment {
    int id;
    Vec2 start;
    Vec2 end;
    double length;
    Vec2 center() const { return (start + end) * 0.5; }
};

struct PathNode {
    int segment_idx;
    bool reverse; 
};

struct Island {
    int id;
    Vec2 center;
    std::vector<int> segment_indices;
    std::vector<PathNode> optimized_path;
};

// --- CONFIGURATION ---
struct Config {
    double field_size = 0.2;
    double platform_speed = 0.25;
    double galvo_mark_speed = 2.5;
    double galvo_jump_speed = 5.0;
    
    double svg_scale_factor = 1.0;
    double target_w = 0.0;
    double target_h = 0.0;
    
    std::string input_file = "";
    std::string html_file = "result.html";
    std::string gcode_file = "output.gcode";

    void print() {
        fmt::print("\n--- CONFIGURATION ---\n");
        fmt::print("Machine Field:    {:.3f} m\n", field_size);
        if (target_w > 0) 
            fmt::print("Normalization:    Fit to {:.3f}m x {:.3f}m\n", target_w, target_h);
        else 
            fmt::print("Scale Factor:     {:.4f}\n", svg_scale_factor);
        fmt::print("Output GCode:     {}\n", gcode_file);
        fmt::print("---------------------\n");
    }
};

// --- MODULE 1: ACCURATE SVG IMPORTER ---
class SVGLoader {
public:
    static std::vector<Segment> load_from_file(const std::string& filepath) {
        std::vector<Segment> segments;
        segments.reserve(100000); 

        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            fmt::print("[Error] Cannot open file: {}\n", filepath);
            return {};
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::string content(size, ' ');
        file.read(&content[0], size);

        // Normalize newlines
        std::replace(content.begin(), content.end(), '\n', ' ');
        std::replace(content.begin(), content.end(), '\r', ' ');

        const char* cursor = content.c_str();
        const char* end = cursor + size;
        int id_counter = 0;

        while (cursor < end) {
            const char* d_pos = strstr(cursor, " d=");
            if (!d_pos) break;

            const char* quote_start = strpbrk(d_pos, "\"'");
            if (!quote_start) break;
            
            char quote_char = *quote_start;
            const char* path_start = quote_start + 1;
            const char* path_end = strchr(path_start, quote_char);
            if (!path_end) break;

            parse_path_data(path_start, path_end, segments, id_counter);
            cursor = path_end + 1;
        }
        
        return segments;
    }

private:
    static double fast_atof(const char*& p, const char* end) {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (p >= end) return 0.0;
        char* next_p;
        double val = std::strtod(p, &next_p);
        if (next_p == p) return 0.0; 
        p = next_p;
        return val;
    }

    static void parse_path_data(const char* p, const char* end, std::vector<Segment>& segs, int& id_counter) {
        Vec2 curr = {0, 0};
        Vec2 start_subpath = {0, 0};
        char last_cmd = 'M';

        while (p < end) {
            while (p < end && (*p == ' ' || *p == ',')) p++;
            if (p >= end) break;

            char cmd = *p;
            if (std::isalpha(cmd)) {
                last_cmd = cmd;
                p++;
            } else {
                if (last_cmd == 'M') cmd = 'L';
                else if (last_cmd == 'm') cmd = 'l';
                else cmd = last_cmd;
            }

            double vals[6];
            auto get_n = [&](int n) { for(int k=0; k<n; k++) vals[k] = fast_atof(p, end); };

            // FIX: Explicit Vec2 construction to satisfy Eigen operator+=
            if (cmd == 'M') { 
                get_n(2); 
                curr = Vec2(vals[0], vals[1]); 
                start_subpath = curr; 
            }
            else if (cmd == 'm') { 
                get_n(2); 
                curr += Vec2(vals[0], vals[1]); 
                start_subpath = curr; 
            }
            else if (cmd == 'L') { 
                get_n(2); 
                Vec2 pt(vals[0], vals[1]); 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'l') { 
                get_n(2); 
                Vec2 pt = curr + Vec2(vals[0], vals[1]); 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'H') { 
                get_n(1); 
                Vec2 pt = curr; pt.x() = vals[0]; 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'h') { 
                get_n(1); 
                Vec2 pt = curr; pt.x() += vals[0]; 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'V') { 
                get_n(1); 
                Vec2 pt = curr; pt.y() = vals[0]; 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'v') { 
                get_n(1); 
                Vec2 pt = curr; pt.y() += vals[0]; 
                add_segment(segs, id_counter, curr, pt); 
                curr = pt; 
            }
            else if (cmd == 'Z' || cmd == 'z') { 
                add_segment(segs, id_counter, curr, start_subpath); 
                curr = start_subpath; 
            }
            else if (cmd == 'C') { 
                get_n(6); 
                tessellate_cubic(segs, id_counter, curr, {vals[0], vals[1]}, {vals[2], vals[3]}, {vals[4], vals[5]}); 
                curr = {vals[4], vals[5]}; 
            }
            else if (cmd == 'c') { 
                get_n(6); 
                tessellate_cubic(segs, id_counter, curr, curr+Vec2(vals[0],vals[1]), curr+Vec2(vals[2],vals[3]), curr+Vec2(vals[4],vals[5])); 
                curr += Vec2(vals[4], vals[5]); 
            }
        }
    }

    static void add_segment(std::vector<Segment>& segs, int& id, Vec2 p1, Vec2 p2) {
        double len = (p2 - p1).norm();
        if (len > 1e-9) segs.push_back({id++, p1, p2, len});
    }

    static void tessellate_cubic(std::vector<Segment>& segs, int& id, Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3) {
        double rough_len = (p1-p0).norm() + (p2-p1).norm() + (p3-p2).norm();
        int steps = std::clamp((int)(rough_len * 1000.0), 16, 128);
        Vec2 prev = p0;
        for (int i = 1; i <= steps; i++) {
            double t = (double)i / steps;
            double u = 1.0 - t;
            Vec2 pos = (u*u*u)*p0 + 3*(u*u)*t*p1 + 3*u*(t*t)*p2 + (t*t*t)*p3;
            add_segment(segs, id, prev, pos); prev = pos;
        }
    }
};

// --- MODULE 2: NORMALIZATION ---
class GeometryUtils {
public:
    static void normalize(std::vector<Segment>& segments, const Config& cfg) {
        if (segments.empty()) return;

        // 1. FLIP Y (Fix up-to-down)
        for(auto& s : segments) {
            s.start.y() = -s.start.y();
            s.end.y()   = -s.end.y();
        }

        // 2. Bounds
        Vec2 min_pt(1e15, 1e15);
        Vec2 max_pt(-1e15, -1e15);

        for (const auto& s : segments) {
            min_pt = min_pt.cwiseMin(s.start.cwiseMin(s.end));
            max_pt = max_pt.cwiseMax(s.start.cwiseMax(s.end));
        }

        double curr_w = max_pt.x() - min_pt.x();
        double curr_h = max_pt.y() - min_pt.y();
        if (curr_w < 1e-9) curr_w = 1.0;
        if (curr_h < 1e-9) curr_h = 1.0;

        // 3. Scale
        double scale = cfg.svg_scale_factor;
        if (cfg.target_w > 0 && cfg.target_h > 0) {
            double sx = cfg.target_w / curr_w;
            double sy = cfg.target_h / curr_h;
            scale = std::min(sx, sy);
            fmt::print("[Process] Normalize: {:.2f}x{:.2f} -> {:.2f}x{:.2f} (Scale {:.4f})\n", 
                       curr_w, curr_h, curr_w*scale, curr_h*scale, scale);
        }

        // 4. Transform
        for (auto& s : segments) {
            s.start = (s.start - min_pt) * scale;
            s.end   = (s.end - min_pt) * scale;
            s.length = (s.end - s.start).norm(); 
        }
    }
};

// --- MODULE 3: CLUSTERING ---
class ClusterEngine {
public:
    static std::vector<Island> partition(const std::vector<Segment>& segments, int k) {
        int n = segments.size();
        if (n == 0) return {};
        if (n < k) k = n;

        std::vector<Vec2> centroids;
        centroids.reserve(k);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, n - 1);
        centroids.push_back(segments[dist(rng)].center());

        std::vector<double> min_dists(n, 1e15);
        
        for (int i = 1; i < k; ++i) {
            double sum_sq = 0.0;
            #pragma omp parallel for reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                double d = (segments[j].center() - centroids.back()).squaredNorm();
                if (d < min_dists[j]) min_dists[j] = d;
                sum_sq += min_dists[j]; 
            }
            std::uniform_real_distribution<double> rand_d(0, sum_sq);
            double target = rand_d(rng);
            double curr = 0.0;
            for(int j=0; j<n; ++j) {
                curr += min_dists[j];
                if(curr >= target) { centroids.push_back(segments[j].center()); break; }
            }
        }

        std::vector<int> assign(n);
        std::vector<Island> islands(k);

        for (int iter = 0; iter < 8; ++iter) {
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                double best = 1e15;
                int idx = -1;
                for (int c = 0; c < k; ++c) {
                    double d = (segments[i].center() - centroids[c]).squaredNorm();
                    if (d < best) { best = d; idx = c; }
                }
                assign[i] = idx;
            }
            std::vector<Vec2> new_cen(k, Vec2::Zero());
            std::vector<double> counts(k, 0);
            for(int i=0; i<n; ++i) { new_cen[assign[i]] += segments[i].center(); counts[assign[i]]++; }
            for(int c=0; c<k; ++c) if(counts[c]>0) centroids[c] = new_cen[c]/counts[c];
        }

        for(int c=0; c<k; ++c) { islands[c].id = c; islands[c].center = centroids[c]; }
        for(int i=0; i<n; ++i) islands[assign[i]].segment_indices.push_back(i);
        islands.erase(std::remove_if(islands.begin(), islands.end(), 
            [](const Island& isl){ return isl.segment_indices.empty(); }), islands.end());
        return islands;
    }
};

// --- MODULE 4: MICRO SOLVER ---
class MicroSolver {
public:
    static void solve(Island& isl, const std::vector<Segment>& all_segs) {
        auto& idxs = isl.segment_indices;
        int n = idxs.size();
        if(n==0) return;

        std::vector<PathNode> path;
        path.reserve(n);
        std::vector<bool> vis(n, false);
        Vec2 curr = isl.center;

        for(int step=0; step<n; ++step) {
            int best_i = -1;
            bool best_rev = false;
            double min_d = 1e15;

            for(int i=0; i<n; ++i) {
                if(vis[i]) continue;
                const auto& s = all_segs[idxs[i]];
                double d1 = (curr - s.start).squaredNorm();
                
                if(d1 < min_d) { min_d = d1; best_i = i; best_rev = false; }
                double d2 = (curr - s.end).squaredNorm();
                if(d2 < min_d) { min_d = d2; best_i = i; best_rev = true; }
            }
            vis[best_i] = true;
            path.push_back({idxs[best_i], best_rev});
            const auto& s = all_segs[idxs[best_i]];
            curr = best_rev ? s.start : s.end;
        }
        isl.optimized_path = path;
    }
};

// --- MODULE 5: PLATFORM SOLVER ---
class PlatformSolver {
public:
    static std::vector<int> solve(const std::vector<Island>& islands) {
        int n = islands.size();
        std::vector<int> path; 
        std::vector<bool> vis(n, false);
        int curr = 0;
        path.push_back(0);
        vis[0] = true;

        for(int i=1; i<n; ++i) {
            double best = 1e15;
            int next = -1;
            for(int j=0; j<n; ++j) {
                if(!vis[j]) {
                    double d = (islands[curr].center - islands[j].center).squaredNorm();
                    if(d < best) { best = d; next = j; }
                }
            }
            vis[next] = true;
            path.push_back(next);
            curr = next;
        }
        return path;
    }
};

// --- MODULE 6: EXPORTERS ---
class Exporter {
public:
    static void export_html(const Config& cfg, const std::vector<Segment>& segments,
                            const std::vector<Island>& islands, const std::vector<int>& plat_order) 
    {
        std::ofstream out(cfg.html_file);
        if(!out.is_open()) return;

        double min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9;
        for(const auto& s : segments) {
            min_x = std::min({min_x, s.start.x(), s.end.x()});
            max_x = std::max({max_x, s.start.x(), s.end.x()});
            min_y = std::min({min_y, s.start.y(), s.end.y()});
            max_y = std::max({max_y, s.start.y(), s.end.y()});
        }
        double pad = std::max(0.01, (max_x - min_x) * 0.05);
        double vb_w = (max_x - min_x) + 2*pad;
        double vb_h = (max_y - min_y) + 2*pad;
        double field = cfg.field_size;

        out << R"(<!DOCTYPE html><html><head><style>
            body { background: #1a1a1a; color: #ccc; font-family: monospace; display: flex; flex-direction: column; align-items: center; }
            svg { border: 1px solid #444; background: #000; }
            .cut { stroke: #ff3366; stroke-width: )" << vb_w/1000.0 << R"(; stroke-linecap: round; }
            .jump { stroke: #555; stroke-width: )" << vb_w/2500.0 << R"(; stroke-dasharray: )" << vb_w/200.0 << R"(; }
            .plat { stroke: #00aaff; stroke-width: )" << vb_w/800.0 << R"(; stroke-dasharray: )" << vb_w/100.0 << R"(; opacity: 0.7; fill: none;}
            .station { fill: none; stroke: #00aaff; stroke-width: )" << vb_w/1500.0 << R"(; opacity: 0.3; }
        </style></head><body><h3>File: )" << cfg.input_file << "</h3>";
        
        out << fmt::format("<svg width='1000' height='1000' viewBox='{} {} {} {}'>",
            min_x - pad, min_y - pad, vb_w, vb_h);

        out << "<path class='plat' d='M";
        for(size_t i=0; i<plat_order.size(); ++i) {
            Vec2 p = islands[plat_order[i]].center;
            out << fmt::format(" {} {}{}", p.x(), p.y(), (i==plat_order.size()-1 ? "'" : " L"));
        }
        out << "/>\n";

        for(int isl_idx : plat_order) {
            const auto& isl = islands[isl_idx];
            out << fmt::format("<rect class='station' x='{}' y='{}' width='{}' height='{}' />\n",
                isl.center.x() - field/2, isl.center.y() - field/2, field, field);
            
            Vec2 head = isl.center;
            for(const auto& node : isl.optimized_path) {
                const auto& seg = segments[node.segment_idx];
                Vec2 start = node.reverse ? seg.end : seg.start;
                Vec2 end   = node.reverse ? seg.start : seg.end;
                out << fmt::format("<line class='jump' x1='{}' y1='{}' x2='{}' y2='{}' />\n", head.x(), head.y(), start.x(), start.y());
                out << fmt::format("<line class='cut' x1='{}' y1='{}' x2='{}' y2='{}' />\n", start.x(), start.y(), end.x(), end.y());
                head = end;
            }
        }
        out << "</svg></body></html>";
    }

    static void export_gcode(const Config& cfg, const std::vector<Segment>& segments,
                             const std::vector<Island>& islands, const std::vector<int>& plat_order)
    {
        std::ofstream out(cfg.gcode_file);
        if(!out.is_open()) {
            fmt::print("[Error] Cannot open output file: {}\n", cfg.gcode_file);
            return;
        }

        out << "; Generated by LaserSpeedPlanner v3.0\n";
        out << "; Input: " << cfg.input_file << "\n";
        out << "G21 ; Units: Millimeters\n";
        out << "G90 ; Absolute Positioning\n";
        
        for(int isl_idx : plat_order) {
            const auto& isl = islands[isl_idx];
            
            out << "\n; --- Station " << isl.id << " ---\n";
            out << fmt::format("G0 X{:.3f} Y{:.3f} F{:.0f} ; Platform\n", 
                isl.center.x() * 1000.0, isl.center.y() * 1000.0, cfg.platform_speed * 60000.0);
            out << "M0 ; Settle\n";

            for(const auto& node : isl.optimized_path) {
                const auto& seg = segments[node.segment_idx];
                Vec2 start = node.reverse ? seg.end : seg.start;
                Vec2 end   = node.reverse ? seg.start : seg.end;

                Vec2 rel_start = start - isl.center;
                Vec2 rel_end   = end - isl.center;

                out << fmt::format("G0 X{:.3f} Y{:.3f} F{:.0f}\n", 
                    rel_start.x()*1000.0, rel_start.y()*1000.0, cfg.galvo_jump_speed * 60000.0);
                
                out << "M3\n";
                out << fmt::format("G1 X{:.3f} Y{:.3f} F{:.0f}\n", 
                    rel_end.x()*1000.0, rel_end.y()*1000.0, cfg.galvo_mark_speed * 60000.0);
                out << "M5\n";
            }
        }
        out << "M2 ; End Program\n";
    }
};

// --- ARGUMENT PARSER ---
double safe_arg(char** argv, int& i, int argc) {
    if (i + 1 >= argc) return 0.0;
    std::string s = argv[++i];
    try { return std::stod(s); } catch(...) { return 0.0; }
}

Config parse_arguments(int argc, char** argv) {
    Config cfg;
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--platform_speed") cfg.platform_speed = safe_arg(argv, i, argc);
        else if (arg == "--galvo_mark_speed") cfg.galvo_mark_speed = safe_arg(argv, i, argc);
        else if (arg == "--galvo_jump_speed") cfg.galvo_jump_speed = safe_arg(argv, i, argc);
        else if (arg == "--galvo_field_size") cfg.field_size = safe_arg(argv, i, argc);
        else if (arg == "--svg_scale_factor") cfg.svg_scale_factor = safe_arg(argv, i, argc);
        else if (arg == "--normalize") {
             cfg.target_w = safe_arg(argv, i, argc);
             cfg.target_h = safe_arg(argv, i, argc);
        }
        else if (arg == "--gcode_output") cfg.gcode_file = argv[++i];
        else if (arg == "--output") cfg.html_file = argv[++i];
        else if (arg[0] != '-') cfg.input_file = arg;
    }
    return cfg;
}

// --- MAIN ---
int main(int argc, char** argv) {
    Config cfg = parse_arguments(argc, argv);
    
    std::cout << "[1/5] Loading..." << std::endl;
    std::vector<Segment> segments;
    if (!cfg.input_file.empty()) segments = SVGLoader::load_from_file(cfg.input_file);
    else { std::cout << "Usage: ./laser_planner input.svg [options]" << std::endl; return 1; }
    
    if (segments.empty()) { std::cerr << "Empty file." << std::endl; return 1; }
    
    GeometryUtils::normalize(segments, cfg);
    cfg.print();
    
    auto start = std::chrono::high_resolution_clock::now();

    // Cluster
    double min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9;
    for(const auto& s : segments) {
        min_x = std::min({min_x, s.start.x(), s.end.x()});
        max_x = std::max({max_x, s.start.x(), s.end.x()});
        min_y = std::min({min_y, s.start.y(), s.end.y()});
        max_y = std::max({max_y, s.start.y(), s.end.y()});
    }
    double area = (max_x - min_x) * (max_y - min_y);
    int k_est = std::max(1, (int)((area / (cfg.field_size * cfg.field_size)) * 1.5)) + 1;
    if (segments.size() > 50000 && k_est < 50) k_est = 50;

    fmt::print("[2/5] Clustering into {} islands...\n", k_est);
    auto islands = ClusterEngine::partition(segments, k_est);

    fmt::print("[3/5] Solving {} Islands...\n", islands.size());
    #pragma omp parallel for
    for(size_t i=0; i<islands.size(); ++i) MicroSolver::solve(islands[i], segments);

    std::cout << "[4/5] Platform TSP..." << std::endl;
    auto order = PlatformSolver::solve(islands);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end-start).count();

    // --- 6. CALCULATE PHYSICAL TIME ---
    double t_plat = 0.0;
    double t_jump = 0.0;
    double t_mark = 0.0;
    
    Vec2 plat_pos(0,0); 

    for (int isl_idx : order) {
        const auto& isl = islands[isl_idx];
        
        double dist = (isl.center - plat_pos).norm();
        t_plat += dist / cfg.platform_speed;
        t_plat += 0.1; // Settle time
        plat_pos = isl.center;

        Vec2 head = isl.center;
        
        for (const auto& node : isl.optimized_path) {
            const auto& seg = segments[node.segment_idx];
            Vec2 start = node.reverse ? seg.end : seg.start;
            
            t_jump += (start - head).norm() / cfg.galvo_jump_speed;
            t_mark += seg.length / cfg.galvo_mark_speed;
            
            head = node.reverse ? seg.start : seg.end;
        }
    }

    double total_physical_time = t_plat + t_jump + t_mark;

    // --- 7. EXPORT & PRINT ---
    Exporter::export_html(cfg, segments, islands, order);
    Exporter::export_gcode(cfg, segments, islands, order);

    fmt::print("\n=== PERFORMANCE REPORT ===\n");
    fmt::print("CPU Compute Time:   {:.2f} ms\n", ms);
    fmt::print("--------------------------\n");
    fmt::print("Platform Travel:    {:.2f} s\n", t_plat);
    fmt::print("Galvo Jumps (OFF):  {:.2f} s\n", t_jump);
    fmt::print("Laser Mark (ON):    {:.2f} s\n", t_mark);
    fmt::print("TOTAL MACHINE TIME: {:.2f} s\n", total_physical_time);
    fmt::print("==========================\n");
    fmt::print("Output HTML:  {}\n", cfg.html_file);
    fmt::print("Output GCode: {}\n", cfg.gcode_file);

    return 0;
}