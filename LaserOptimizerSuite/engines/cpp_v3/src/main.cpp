/**
 * LASER SPEED PLANNER v4.0 (C++17)
 * 
 * FIXES:
 * - G-Code Optimization: Removes redundant Jumps (Zero-distance moves).
 * - Viewer Compatibility: Re-added "; Platform" tag so the viewer renders correctly.
 * - Precision: High-resolution float handling.
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
#include <cstdlib>
#include <omp.h>
#include <Eigen/Dense>
#include <fmt/core.h>

// --- CORE TYPES ---
using Vec2 = Eigen::Vector2f;
using Float = float; 

struct Segment {
    int id;
    Vec2 start;
    Vec2 end;
    Float length;
    Vec2 center() const { return (start + end) * 0.5f; }
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
    Float field_size = 0.2f;
    Float platform_speed = 0.25f;
    Float galvo_mark_speed = 2.5f;
    Float galvo_jump_speed = 5.0f;
    
    Float svg_scale_factor = 1.0f;
    Float target_w = 0.0f;
    Float target_h = 0.0f;
    Float segment_len_mm = 0.05f; 
    
    int tile_n = 1; 

    std::string input_file = "";
    std::string html_file = "result.html";
    std::string gcode_file = "output.gcode";

    void print() {
        fmt::print("\n--- CONFIGURATION ---\n");
        fmt::print("Machine Field:    {:.3f} m\n", field_size);
        if (target_w > 0) 
            fmt::print("Normalization:    Fit to {:.3f}m x {:.3f}m\n", target_w, target_h);
        fmt::print("Target Accuracy:  {:.4f} mm\n", segment_len_mm);
        if (tile_n > 1)
            fmt::print("Tiling:           {} x {} Grid\n", tile_n, tile_n);
        fmt::print("---------------------\n");
    }
};

// --- MODULE 1: TWO-PASS LOADER ---
class SVGLoader {
    static const size_t MAX_TOTAL_SEGMENTS = 100000000; 

public:
    static std::vector<Segment> load_and_normalize(const Config& cfg) {
        std::string filepath = cfg.input_file;
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            fmt::print("[Error] Cannot open file: {}\n", filepath);
            return {};
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::string content(size, ' ');
        file.read(&content[0], size);
        
        std::replace(content.begin(), content.end(), '\n', ' ');
        std::replace(content.begin(), content.end(), '\r', ' ');
        std::replace(content.begin(), content.end(), ',', ' ');

        fmt::print("[1/5] Scanning bounds...\n");
        Vec2 min_pt(1e15f, 1e15f), max_pt(-1e15f, -1e15f);
        scan_bounds(content, min_pt, max_pt);
        
        Float raw_w = max_pt.x() - min_pt.x();
        Float raw_h = max_pt.y() - min_pt.y();
        if (raw_w < 1e-9f) raw_w = 1.0f;
        if (raw_h < 1e-9f) raw_h = 1.0f;

        Float scale = cfg.svg_scale_factor;
        if (cfg.target_w > 0 && cfg.target_h > 0) {
            Float sx = cfg.target_w / raw_w;
            Float sy = cfg.target_h / raw_h;
            scale = std::min(sx, sy);
        }

        Float target_m = cfg.segment_len_mm / 1000.0f;
        Float raw_threshold = target_m / scale;
        
        fmt::print("      Raw Size: {:.2f} x {:.2f}\n", raw_w, raw_h);
        fmt::print("      Scale: {:.6f}\n", scale);

        std::vector<Segment> segments;
        segments.reserve(2000000); 
        int id_counter = 0;
        parse_content(content, segments, id_counter, raw_threshold, min_pt, scale);

        fmt::print("      Generated {} segments.\n", segments.size());
        return segments;
    }

private:
    static Float fast_atof(const char*& p, const char* end) {
        while (p < end && *p == ' ') p++;
        if (p >= end) return 0.0f;
        char* next_p;
        Float val = std::strtof(p, &next_p); 
        if (next_p == p) return 0.0f; 
        p = next_p;
        return val;
    }

    static void scan_bounds(const std::string& content, Vec2& min_pt, Vec2& max_pt) {
        const char* p = content.c_str();
        const char* end = p + content.size();
        while (p < end) {
            const char* d_pos = strstr(p, " d=");
            if (!d_pos) break;
            const char* start = strpbrk(d_pos, "\"'");
            if (!start) break;
            const char* path_start = start + 1;
            const char* path_end = strchr(path_start, *start);
            if (!path_end) break;
            run_path_state_machine(path_start, path_end, nullptr, 0, 0, &min_pt, &max_pt);
            p = path_end + 1;
        }
    }

    static void parse_content(const std::string& content, std::vector<Segment>& segs, int& id, Float raw_thresh, Vec2 min_pt, Float scale) {
        const char* p = content.c_str();
        const char* end = p + content.size();
        while (p < end) {
            if (segs.size() >= MAX_TOTAL_SEGMENTS) return;
            const char* d_pos = strstr(p, " d=");
            if (!d_pos) break;
            const char* start = strpbrk(d_pos, "\"'");
            if (!start) break;
            const char* path_start = start + 1;
            const char* path_end = strchr(path_start, *start);
            if (!path_end) break;
            run_path_state_machine(path_start, path_end, &segs, id, raw_thresh, &min_pt, nullptr, scale);
            p = path_end + 1;
        }
    }

    static void run_path_state_machine(const char* p, const char* end, 
                                       std::vector<Segment>* segs, int id_ref, Float raw_thresh,
                                       Vec2* min_out, Vec2* max_out, Float scale = 1.0f) 
    {
        Vec2 curr(0,0), start_subpath(0,0);
        char last_cmd = 'M';
        int local_id = id_ref; 

        auto update_bounds = [&](Vec2 pt) {
            if (min_out) {
                min_out->x() = std::min(min_out->x(), pt.x());
                min_out->y() = std::min(min_out->y(), pt.y());
                if (max_out) {
                    max_out->x() = std::max(max_out->x(), pt.x());
                    max_out->y() = std::max(max_out->y(), pt.y());
                }
            }
        };

        while (p < end) {
            if (segs && segs->size() >= MAX_TOTAL_SEGMENTS) break;
            while (p < end && *p == ' ') p++;
            if (p >= end) break;

            char cmd = *p;
            if (std::isalpha(cmd)) { last_cmd = cmd; p++; }
            else {
                if (last_cmd == 'M') cmd = 'L';
                else if (last_cmd == 'm') cmd = 'l';
                else cmd = last_cmd;
            }

            Float vals[6];
            auto get_n = [&](int n) { for(int k=0; k<n; k++) vals[k] = fast_atof(p, end); };

            if (cmd == 'M') { get_n(2); curr = Vec2(vals[0], vals[1]); start_subpath = curr; update_bounds(curr); }
            else if (cmd == 'm') { get_n(2); curr += Vec2(vals[0], vals[1]); start_subpath = curr; update_bounds(curr); }
            else if (cmd == 'L') { 
                get_n(2); Vec2 pt(vals[0], vals[1]); 
                if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale);
                curr = pt; update_bounds(curr);
            }
            else if (cmd == 'l') { 
                get_n(2); Vec2 pt = curr + Vec2(vals[0], vals[1]); 
                if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale);
                curr = pt; update_bounds(curr);
            }
            else if (cmd == 'H') { get_n(1); Vec2 pt = curr; pt.x() = vals[0]; if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale); curr = pt; update_bounds(curr); }
            else if (cmd == 'h') { get_n(1); Vec2 pt = curr; pt.x() += vals[0]; if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale); curr = pt; update_bounds(curr); }
            else if (cmd == 'V') { get_n(1); Vec2 pt = curr; pt.y() = vals[0]; if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale); curr = pt; update_bounds(curr); }
            else if (cmd == 'v') { get_n(1); Vec2 pt = curr; pt.y() += vals[0]; if (segs) add_transformed(*segs, local_id, curr, pt, *min_out, scale); curr = pt; update_bounds(curr); }
            else if (cmd == 'Z' || cmd == 'z') { if (segs) add_transformed(*segs, local_id, curr, start_subpath, *min_out, scale); curr = start_subpath; }
            else if (cmd == 'C') { 
                get_n(6); 
                Vec2 p1(vals[0], vals[1]), p2(vals[2], vals[3]), p3(vals[4], vals[5]);
                if (segs) tessellate_cubic(*segs, local_id, curr, p1, p2, p3, raw_thresh, *min_out, scale);
                else { update_bounds(p1); update_bounds(p2); update_bounds(p3); }
                curr = p3;
            }
            else if (cmd == 'c') { 
                get_n(6); 
                Vec2 p1 = curr+Vec2(vals[0],vals[1]), p2 = curr+Vec2(vals[2],vals[3]), p3 = curr+Vec2(vals[4],vals[5]);
                if (segs) tessellate_cubic(*segs, local_id, curr, p1, p2, p3, raw_thresh, *min_out, scale);
                else { update_bounds(p1); update_bounds(p2); update_bounds(p3); }
                curr = p3;
            }
        }
        if (segs) const_cast<int&>(id_ref) = local_id;
    }

    static void add_transformed(std::vector<Segment>& segs, int& id, Vec2 raw_p1, Vec2 raw_p2, Vec2 offset, Float scale) {
        Float len = (raw_p2 - raw_p1).norm();
        if (len < 1e-9f) return;
        
        // 1. Normalize
        Vec2 p1 = (raw_p1 - offset) * scale;
        Vec2 p2 = (raw_p2 - offset) * scale;
        
        // 2. Y-Flip for MACHINE COORDINATES (Y+ Up)
        p1.y() = -p1.y();
        p2.y() = -p2.y();
        
        segs.push_back({id++, p1, p2, (p2-p1).norm()});
    }

    static void tessellate_cubic(std::vector<Segment>& segs, int& id, 
                                 Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, 
                                 Float raw_thresh, Vec2 offset, Float scale) 
    {
        Float rough_len = (p1-p0).norm() + (p2-p1).norm() + (p3-p2).norm();
        int steps = std::max(2, (int)(rough_len / raw_thresh));
        steps = std::min(steps, 1000); 

        Vec2 prev = p0;
        for (int i = 1; i <= steps; i++) {
            Float t = (Float)i / steps;
            Float u = 1.0f - t;
            Vec2 pos = (u*u*u)*p0 + 3*(u*u)*t*p1 + 3*u*(t*t)*p2 + (t*t*t)*p3;
            add_transformed(segs, id, prev, pos, offset, scale);
            prev = pos;
        }
    }
};

// --- MODULE 1.5: TILER ---
class Tiler {
public:
    static void tile(std::vector<Segment>& segments, int n_tile, Float spacing_m) {
        if (n_tile <= 1) return;
        
        Vec2 min_pt(1e15f, 1e15f), max_pt(-1e15f, -1e15f);
        for(const auto& s : segments) {
            min_pt = min_pt.cwiseMin(s.start.cwiseMin(s.end));
            max_pt = max_pt.cwiseMax(s.start.cwiseMax(s.end));
        }
        Float w = max_pt.x() - min_pt.x();
        Float h = max_pt.y() - min_pt.y();
        
        fmt::print("[Proc] Tiling {}x{} grid (Spacing {:.3f}m)...\n", n_tile, n_tile, spacing_m);
        
        std::vector<Segment> tiles;
        tiles.reserve(segments.size() * n_tile * n_tile);
        
        int id_counter = 0;
        for(int iy=0; iy<n_tile; ++iy) {
            for(int ix=0; ix<n_tile; ++ix) {
                Vec2 offset(ix * (w + spacing_m), iy * (h + spacing_m));
                for(const auto& s : segments) {
                    Segment ns = s;
                    ns.id = id_counter++;
                    ns.start += offset;
                    ns.end += offset;
                    tiles.push_back(ns);
                }
            }
        }
        segments = tiles;
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

        std::vector<Float> min_dists(n, 1e15f);
        for (int i = 1; i < k; ++i) {
            double sum_sq = 0.0;
            #pragma omp parallel for reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                Float d = (segments[j].center() - centroids.back()).squaredNorm();
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
                Float best = 1e15f;
                int idx = -1;
                for (int c = 0; c < k; ++c) {
                    Float d = (segments[i].center() - centroids[c]).squaredNorm();
                    if (d < best) { best = d; idx = c; }
                }
                assign[i] = idx;
            }
            std::vector<Vec2> new_cen(k, Vec2::Zero());
            std::vector<Float> counts(k, 0);
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
            Float min_d = 1e15f;

            for(int i=0; i<n; ++i) {
                if(vis[i]) continue;
                const auto& s = all_segs[idxs[i]];
                Float d1 = (curr - s.start).squaredNorm();
                if(d1 < min_d) { min_d = d1; best_i = i; best_rev = false; }
                Float d2 = (curr - s.end).squaredNorm();
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
            Float best = 1e15f;
            int next = -1;
            for(int j=0; j<n; ++j) {
                if(!vis[j]) {
                    Float d = (islands[curr].center - islands[j].center).squaredNorm();
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
class MetricsExporter {
public:
    static void save(const std::string& filename, double ms, double t_plat, double t_jump, double t_mark, int segs, int islands, double total_len) {
        double res = (segs > 0) ? (total_len * 1000.0 / segs) : 0.0;
        std::ofstream out(filename);
        out << "{\n  \"engine\": \"cpp_v3\",\n  \"vector_count\": " << segs 
            << ",\n  \"island_count\": " << islands << ",\n  \"geo_resolution_mm\": " << res
            << ",\n  \"time_compute_sec\": " << ms/1000.0 << ",\n  \"time_total_machine_sec\": " << (t_plat+t_jump+t_mark)
            << ",\n  \"time_mark_sec\": " << t_mark << ",\n  \"time_platform_sec\": " << t_plat 
            << ",\n  \"time_jump_sec\": " << t_jump << "\n}\n";
    }
};

class Exporter {
public:
    static void export_html(const Config& cfg, const std::vector<Segment>& segments, const std::vector<Island>& islands, const std::vector<int>& order) {
        std::ofstream out(cfg.html_file);
        if(!out.is_open()) return;

        Float min_x=1e9f, max_x=-1e9f, min_y=1e9f, max_y=-1e9f;
        for(const auto& s : segments) {
            min_x=std::min({min_x,s.start.x(),s.end.x()}); max_x=std::max({max_x,s.start.x(),s.end.x()});
            min_y=std::min({min_y,s.start.y(),s.end.y()}); max_y=std::max({max_y,s.start.y(),s.end.y()});
        }
        
        Float w_data = max_x - min_x;
        Float h_data = max_y - min_y;
        Float pad = std::max(0.01f, w_data * 0.05f);
        Float w_view = w_data + 2*pad;
        Float h_view = h_data + 2*pad;
        Float field = cfg.field_size;

        out << R"(<html><body style="background:#111; color:#888; font-family:sans-serif;">)";
        out << "<h3>" << cfg.input_file << "</h3>";
        
        out << fmt::format(R"(<svg width='100%' height='90vh' viewBox='{} {} {} {}' style="border:1px solid #444">)",
            min_x - pad, 
            -max_y - pad, 
            w_view, 
            h_view);
            
        out << R"(<g transform=\"scale(1, -1)\">)";

        // Platform Path
        out << "<path fill='none' stroke='#00aaff' stroke-width='" << w_view/800.0 << "' stroke-dasharray='" << w_view/100.0 << "' d='M";
        for(size_t i=0; i<order.size(); ++i) {
            Vec2 p = islands[order[i]].center;
            out << " " << p.x() << " " << -p.y() << (i==order.size()-1 ? "'" : " L");
        }
        out << "/>\n";

        for(int idx : order) {
            const auto& isl = islands[idx];
            Float cx = isl.center.x(), cy = -isl.center.y();
            out << fmt::format("<rect x='{}' y='{}' width='{}' height='{}' fill='none' stroke='#00aaff' stroke-width='{}' opacity='0.3' />\n",
                cx - field/2, cy - field/2, field, field, w_view/1500.0);
            
            Vec2 head = isl.center;
            for(const auto& node : isl.optimized_path) {
                const auto& seg = segments[node.segment_idx];
                Vec2 start = node.reverse ? seg.end : seg.start;
                Vec2 end   = node.reverse ? seg.start : seg.end;
                
                // --- OPTIMIZATION: REMOVE ZERO-LENGTH JUMPS ---
                // If head matches start (within tolerance), skip drawing the jump
                Float jump_dist = (start - head).norm();
                if (jump_dist > 0.0001f) {
                    out << fmt::format("<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#555' stroke-width='{}' stroke-dasharray='{}' />\n",
                        head.x(), -head.y(), start.x(), -start.y(), w_view/2500.0, w_view/200.0);
                }
                
                out << fmt::format("<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#ff3366' stroke-width='{}' />\n",
                    start.x(), -start.y(), end.x(), -end.y(), w_view/1000.0);
                    
                head = end;
            }
        }
        out << "</g></svg></body></html>";
    }

    static void export_gcode(const Config& cfg, const std::vector<Segment>& segments, const std::vector<Island>& islands, const std::vector<int>& order, double total_time) {
        std::ofstream out(cfg.gcode_file);
        out << "; Generated by LaserSpeedPlanner v4.0\n";
        out << "; Total Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
        out << "G21\nG90\n";
        
        for(int idx : order) {
            auto& isl = islands[idx];
            
            // --- FIX: ADD ; Platform TAG ---
            // This is critical for the Python viewer to distinguish Stage vs Galvo
            out << fmt::format("\n; Station {}\nG0 X{:.3f} Y{:.3f} F{:.0f} ; Platform\nM0\n", 
                isl.id, isl.center.x()*1000, isl.center.y()*1000, cfg.platform_speed*60000);
            
            Vec2 head = isl.center; // Track logical head pos relative to island
            
            for(auto& n : isl.optimized_path) {
                auto& s = segments[n.segment_idx];
                Vec2 start = n.reverse ? s.end : s.start;
                Vec2 end = n.reverse ? s.start : s.end;
                Vec2 rs = start - isl.center, re = end - isl.center;
                
                // --- OPTIMIZATION: REMOVE ZERO-LENGTH JUMPS ---
                // Only write G0 if distance > epsilon
                if ((start - head).norm() > 0.00001f) {
                    out << fmt::format("G0 X{:.3f} Y{:.3f}\n", rs.x()*1000, rs.y()*1000);
                }
                
                out << "M3\n";
                out << fmt::format("G1 X{:.3f} Y{:.3f}\n", re.x()*1000, re.y()*1000);
                out << "M5\n";
                
                head = end;
            }
        }
        out << "M2\n";
    }
};

// --- MAIN ---
Float safe_arg_float(char** argv, int& i, int argc) {
    if (i + 1 >= argc) return 0.0f;
    try { return std::stof(argv[++i]); } catch(...) { return 0.0f; }
}

int safe_arg_int(char** argv, int& i, int argc) {
    if (i + 1 >= argc) return 1;
    try { return std::stoi(argv[++i]); } catch(...) { return 1; }
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for(int i=1; i<argc; ++i) {
        std::string s = argv[i];
        if (s == "--platform_speed") cfg.platform_speed = safe_arg_float(argv, i, argc);
        else if (s == "--galvo_mark_speed") cfg.galvo_mark_speed = safe_arg_float(argv, i, argc);
        else if (s == "--galvo_jump_speed") cfg.galvo_jump_speed = safe_arg_float(argv, i, argc);
        else if (s == "--galvo_field_size") cfg.field_size = safe_arg_float(argv, i, argc);
        else if (s == "--svg_scale_factor") cfg.svg_scale_factor = safe_arg_float(argv, i, argc);
        else if (s == "--segment_len") cfg.segment_len_mm = safe_arg_float(argv, i, argc);
        else if (s == "--normalize") { cfg.target_w = safe_arg_float(argv, i, argc); cfg.target_h = safe_arg_float(argv, i, argc); }
        else if (s == "--tile") cfg.tile_n = safe_arg_int(argv, i, argc);
        else if (s == "--gcode_output") cfg.gcode_file = argv[++i];
        else if (s == "--output") cfg.html_file = argv[++i];
        else if (s[0] != '-') cfg.input_file = s;
    }
    return cfg;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    if(cfg.input_file.empty()) return 1;

    auto segments = SVGLoader::load_and_normalize(cfg);
    if(segments.empty()) return 1;
    cfg.print();

    if(cfg.tile_n > 1) Tiler::tile(segments, cfg.tile_n, 0.02f);

    auto start = std::chrono::high_resolution_clock::now();

    int k_est = std::max(1, (int)(segments.size() / 1000));
    auto islands = ClusterEngine::partition(segments, k_est);

    #pragma omp parallel for
    for(size_t i=0; i<islands.size(); ++i) MicroSolver::solve(islands[i], segments);
    auto order = PlatformSolver::solve(islands);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end-start).count();

    double t_plat=0, t_jump=0, t_mark=0, len=0;
    Vec2 p_pos(0,0);
    for(int idx : order) {
        auto& isl = islands[idx];
        t_plat += (isl.center - p_pos).norm() / cfg.platform_speed + 0.1; p_pos = isl.center;
        Vec2 head = isl.center;
        for(auto& n : isl.optimized_path) {
            auto& s = segments[n.segment_idx];
            len += s.length;
            Vec2 start = n.reverse ? s.end : s.start;
            t_jump += (start - head).norm() / cfg.galvo_jump_speed;
            t_mark += s.length / cfg.galvo_mark_speed;
            head = n.reverse ? s.start : s.end;
        }
    }
    double total_phys = t_plat + t_jump + t_mark;

    Exporter::export_html(cfg, segments, islands, order);
    Exporter::export_gcode(cfg, segments, islands, order, total_phys);
    MetricsExporter::save(cfg.gcode_file+".json", ms, t_plat, t_jump, t_mark, segments.size(), islands.size(), len);

    fmt::print("\n=== PERFORMANCE REPORT ===\n");
    fmt::print("CPU Compute:      {:.2f} ms\n", ms);
    fmt::print("Machine Work:     {:.2f} s\n", total_phys);
    fmt::print("Output GCode:     {}\n", cfg.gcode_file);
    fmt::print("Output HTML:      {}\n", cfg.html_file);
    fmt::print("==========================\n");

    return 0;
}