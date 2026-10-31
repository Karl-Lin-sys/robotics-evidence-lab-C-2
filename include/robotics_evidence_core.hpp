#ifndef ROBOTICS_EVIDENCE_CORE_HPP
#define ROBOTICS_EVIDENCE_CORE_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace robotics_evidence {

// ============================================================================
// 1. NIST CSF 2.0 (GOVERN / IDENTIFY / PROTECT / DETECT):
//    Taiwan PDPA (個人資料保護法), Copyright Act (著作權法) & Trade Secrets Guard
// ============================================================================
class TaiwanComplianceGuard {
public:
    struct AuditReport {
        bool pii_or_secret_detected{false};
        std::size_t redaction_count{0};
        std::string sanitized_text;
    };

    /**
     * Scans and redacts Taiwanese phone numbers, email addresses, and sensitive
     * contact identifiers in O(L) time before logging or external repository export.
     *
     * Complexity:
     *   - Time: O(L) where L is the input string length.
     *   - Space: O(L) for the sanitized output buffer.
     */
    static AuditReport audit_and_redact(const std::string& input) {
        static const std::array<std::regex, 3> kPatterns = {
            std::regex(R"(\b09\d{2}[-\s]?\d{3}[-\s]?\d{3}\b)"),
            std::regex(R"(\b0[2-8][-\s]?\d{3,4}[-\s]?\d{4}\b)"),
            std::regex(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
        };

        AuditReport report{false, 0, input};
        for (const auto& pat : kPatterns) {
            auto begin = std::sregex_iterator(
                report.sanitized_text.begin(), report.sanitized_text.end(), pat);
            auto end = std::sregex_iterator();
            std::size_t matches = static_cast<std::size_t>(std::distance(begin, end));
            if (matches > 0) {
                report.pii_or_secret_detected = true;
                report.redaction_count += matches;
                report.sanitized_text = std::regex_replace(
                    report.sanitized_text, pat, "[REDACTED_TW_COMPLIANCE]");
            }
        }
        return report;
    }
};

// ============================================================================
// 2. Perception & SLAM: Continuous-Time SE(3) IMU-Assisted Point Cloud Deskew
//    & Block-Temporal Split Validator (Anti-Leakage Guard)
// ============================================================================
struct Point3DTimestamped {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double intensity_or_thermal{0.0}; // LiDAR reflectivity or calibrated LWIR temperature
    double relative_time_s{0.0};      // Time offset within scan [0, dt_scan]
};

struct ImuMeasurement {
    double ax{0.0}, ay{0.0}, az{0.0}; // Linear acceleration (m/s^2, gravity-compensated)
    double wx{0.0}, wy{0.0}, wz{0.0}; // Angular velocity (rad/s)
};

class MultiModalPerceptionProcessor {
public:
    /**
     * Deskews a 3D LiDAR/RGB-D/LWIR point cloud using first-order SE(3) exponential map
     * integration from high-rate IMU measurements to eliminate motion distortion bias.
     *
     * Complexity:
     *   - Time: O(N) where N is the number of 3D points.
     *   - Space: O(N) contiguous output vector (RAII managed, zero memory leaks).
     */
    static std::vector<Point3DTimestamped> deskew_point_cloud_se3(
        const std::vector<Point3DTimestamped>& raw_points,
        const ImuMeasurement& imu_state) {
        std::vector<Point3DTimestamped> deskewed;
        deskewed.reserve(raw_points.size());

        for (const auto& pt : raw_points) {
            const double dt = pt.relative_time_s;
            // Small-angle SO(3) rotation vector theta = omega * dt
            const double rx = imu_state.wx * dt;
            const double ry = imu_state.wy * dt;
            const double rz = imu_state.wz * dt;

            // Translation from constant acceleration model: p_trans = 0.5 * a * dt^2
            const double tx = 0.5 * imu_state.ax * dt * dt;
            const double ty = 0.5 * imu_state.ay * dt * dt;
            const double tz = 0.5 * imu_state.az * dt * dt;

            // Apply (I + [r]_x) * p + t
            Point3DTimestamped out{};
            out.x = pt.x + (ry * pt.z - rz * pt.y) + tx;
            out.y = pt.y + (rz * pt.x - rx * pt.z) + ty;
            out.z = pt.z + (rx * pt.y - ry * pt.x) + tz;
            out.intensity_or_thermal = pt.intensity_or_thermal;
            out.relative_time_s = 0.0; // Deskewed to scan start reference frame
            deskewed.push_back(out);
        }
        return deskewed;
    }

    /**
     * Validates that training and evaluation trajectory timestamps satisfy a strict
     * temporal decorrelation buffer (tau_decorr) to prevent spatio-temporal data leakage.
     *
     * Complexity:
     *   - Time: O(N log N + M log M) where N, M are train and eval sample counts.
     *   - Space: O(N + M) for sorted timestamp copies.
     */
    static bool verify_block_temporal_split(
        std::vector<double> train_timestamps_s,
        std::vector<double> eval_timestamps_s,
        double min_gap_s = 5.0) {
        if (train_timestamps_s.empty() || eval_timestamps_s.empty()) {
            return true;
        }
        std::sort(train_timestamps_s.begin(), train_timestamps_s.end());
        std::sort(eval_timestamps_s.begin(), eval_timestamps_s.end());

        std::size_t i = 0;
        std::size_t j = 0;
        while (i < train_timestamps_s.size() && j < eval_timestamps_s.size()) {
            if (std::abs(train_timestamps_s[i] - eval_timestamps_s[j]) < min_gap_s) {
                return false; // Data leakage detected across adjacent ROS 2 bag frames
            }
            if (train_timestamps_s[i] < eval_timestamps_s[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        return true;
    }
};

// ============================================================================
// 3. NIST CSF 2.0 (PROTECT / DETECT / RESPOND / RECOVER):
//    Whole-Body Control, Power/Battery & EtherCAT Real-Time Hardware Governor
// ============================================================================
struct SafetyGovernanceConfig {
    static constexpr double kMaxJointTorqueNm = 85.0;
    static constexpr double kMaxThermalCelsius = 75.0;
    static constexpr double kRecoveryThermalMarginC = 10.0;
    static constexpr double kMinBatteryVoltageV = 42.0; // 12S Li-Ion safe cutoff
    static constexpr std::chrono::milliseconds kMaxSensorWatchdog{50};
};

struct EtherCatJointTelemetry {
    uint64_t sequence_id{0};
    std::chrono::steady_clock::time_point timestamp{};
    double requested_torque_nm{0.0};
    double board_temp_c{25.0};
    double bus_voltage_v{48.0};
    uint32_t hmac_tag{0};
};

class NistWholeBodySafetyGovernor {
public:
    /**
     * Validates telemetry integrity, sequence monotonicity, watchdog latency,
     * battery brownout voltage, thermal dissipation limits, and whole-body torque
     * saturation in O(1) time.
     *
     * Complexity:
     *   - Time: O(1) deterministic execution per 1 kHz EtherCAT cycle.
     *   - Space: O(1) auxiliary memory.
     */
    bool process_and_actuate(
        const EtherCatJointTelemetry& pkt,
        uint32_t expected_hmac,
        std::chrono::steady_clock::time_point now) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        // [PROTECT - PR.AA / PR.DS]: Constant-time HMAC verification & anti-replay guard
        if (!verify_constant_time(pkt.hmac_tag, expected_hmac) ||
            pkt.sequence_id <= last_sequence_id_) {
            engage_safe_torque_off();
            return false;
        }

        // [DETECT - DE.CM]: Continuous physical, thermal, voltage, and timing anomaly detection
        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - pkt.timestamp);
        if (age < std::chrono::milliseconds::zero() ||
            age > SafetyGovernanceConfig::kMaxSensorWatchdog ||
            !std::isfinite(pkt.requested_torque_nm) ||
            std::abs(pkt.requested_torque_nm) > SafetyGovernanceConfig::kMaxJointTorqueNm ||
            pkt.board_temp_c > SafetyGovernanceConfig::kMaxThermalCelsius ||
            pkt.bus_voltage_v < SafetyGovernanceConfig::kMinBatteryVoltageV) {
            engage_safe_torque_off();
            return false;
        }

        if (in_safe_stop_) {
            return false;
        }

        last_sequence_id_ = pkt.sequence_id;
        commanded_torque_nm_ = std::clamp(
            pkt.requested_torque_nm,
            -SafetyGovernanceConfig::kMaxJointTorqueNm,
            SafetyGovernanceConfig::kMaxJointTorqueNm);
        return true;
    }

    /**
     * [RECOVER - RC.RP]: Restores operational readiness only when thermal & voltage
     * safety margins are satisfied.
     *
     * Complexity: O(1) time and space.
     */
    bool attempt_recovery(double current_temp_c, double current_voltage_v = 48.0) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (in_safe_stop_ &&
            current_temp_c <= (SafetyGovernanceConfig::kMaxThermalCelsius -
                               SafetyGovernanceConfig::kRecoveryThermalMarginC) &&
            current_voltage_v >= SafetyGovernanceConfig::kMinBatteryVoltageV) {
            in_safe_stop_ = false;
            commanded_torque_nm_ = 0.0;
            return true;
        }
        return !in_safe_stop_;
    }

    bool is_in_safe_stop() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return in_safe_stop_;
    }

    double commanded_torque_nm() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return commanded_torque_nm_;
    }

private:
    static bool verify_constant_time(uint32_t a, uint32_t b) noexcept {
        volatile uint32_t diff = a ^ b;
        return diff == 0U;
    }

    // [RESPOND - RS.MI]: Engage Safe-Torque-Off (STO) immediately
    void engage_safe_torque_off() noexcept {
        in_safe_stop_ = true;
        commanded_torque_nm_ = 0.0;
    }

    mutable std::mutex mutex_;
    uint64_t last_sequence_id_{0};
    double commanded_torque_nm_{0.0};
    bool in_safe_stop_{false};
};

// ============================================================================
// 4. Class-Imbalance Bias Elimination: Logit-Adjusted Balanced Loss
//    (Menon et al., NeurIPS / ICLR - Fisher consistent for balanced error)
// ============================================================================
class LogitAdjustedBalancedLoss {
public:
    /**
     * Computes numerically stable Logit-Adjusted Cross-Entropy Loss for a single sample
     * to eliminate class imbalance bias in multi-modal obstacle detection.
     *
     * Complexity:
     *   - Time: O(C) per sample where C is the number of classes.
     *   - Space: O(1) auxiliary space (in-place LogSumExp).
     */
    static double forward_single(
        const std::vector<double>& logits,
        const std::vector<double>& class_priors,
        std::size_t target_class,
        double tau = 1.0) {
        if (logits.empty() || logits.size() != class_priors.size() ||
            target_class >= logits.size()) {
            throw std::invalid_argument("Invalid logits, priors, or target_class dimensions.");
        }

        const std::size_t num_classes = logits.size();
        double max_adj = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < num_classes; ++c) {
            const double prior = std::max(class_priors[c], 1e-12);
            const double adj = logits[c] + tau * std::log(prior);
            if (adj > max_adj) {
                max_adj = adj;
            }
        }

        double sum_exp = 0.0;
        for (std::size_t c = 0; c < num_classes; ++c) {
            const double prior = std::max(class_priors[c], 1e-12);
            const double adj = logits[c] + tau * std::log(prior);
            sum_exp += std::exp(adj - max_adj);
        }

        const double target_prior = std::max(class_priors[target_class], 1e-12);
        const double target_adj = logits[target_class] + tau * std::log(target_prior);
        return (max_adj + std::log(sum_exp)) - target_adj;
    }
};

} // namespace robotics_evidence

#endif // ROBOTICS_EVIDENCE_CORE_HPP
