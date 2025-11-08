// des_sim.cpp
// Compile: g++ -std=c++17 des_sim.cpp -o des_sim
// Example runs:
// ./des_sim --lambda 0.9 --mu 1.0 --maxServed 20000 --warmup 1000 --reps 10 --seed 12345 --queueCap -1 --term served --outdir .
// ./des_sim --lambda 0.9 --mu 1.0 --horizonT 20000 --warmup 1000 --reps 10 --seed 12345 --queueCap 50 --term time --outdir .
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <queue>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <string>
#include <tuple>
#include <algorithm>

using namespace std;

/* -------------------------
Event definitions
------------------------- */
enum class EventType { ARRIVAL, DEPARTURE };

struct Event {
    double time;
    EventType type;
    // priority_queue in C++ is max-heap, so reverse comparison for earliest-first
    bool operator<(const Event& other) const {
        return time > other.time;
    }
};

/* -------------------------
Termination mode
------------------------- */
enum class TerminationMode { BY_SERVED, BY_TIME };

/* -------------------------
Scenario options
- step change in lambda at time changeTime from lambda1 -> lambda2
- finite queue capacity (queueCap >= 0); queueCap == -1 means infinite
- deterministic service (service time = 1/mu)
   ------------------------- */
struct Scenario {
    bool stepChange = false;
    double changeTime = 0.0;
    double lambda2 = 0.0; // lambda after change time
    bool finiteQueue = false;
    int queueCap = -1; // -1 infinite
    bool deterministicService = false;
};

/* -------------------------
Params, State, Stats
------------------------- */
struct Params {
    double lambda = 0.9;
    double mu = 1.0;
    double horizonT = 20000.0;
    int maxServed = 20000;
    int warmup = 0;            // number of completed customers to skip for stats
    TerminationMode termMode = TerminationMode::BY_SERVED;
    int reps = 10;
    unsigned long long seed = 0;
    int queueCap = -1;         // -1 infinite
    bool deterministicService = false;
    // scenario
    Scenario scenario;
    string outdir = ".";
};

struct State {
    int numInSystem = 0; // in service + in queue
    double lastEventTime = 0.0;
    double areaNumInQueue = 0.0; // integral over time of queue length (excluding in-service)
    double areaServerBusy = 0.0; // integral over time of server busy (0/1)
};

struct Stats {
    int servedTotal = 0;      // total departures (including warmup)
    int servedForStats = 0;   // departures counted after warmup
    int dropped = 0;          // due to finite queue
    double totalDelay = 0.0;  // sum of delays for counted customers
    vector<double> delays;    // delays for counted customers (optional, for analysis)
    double runTime = 0.0;     // simulation time observed (lastEventTime)
};

/* -------------------------
RNG wrapper
------------------------- */
struct RNG {
    std::mt19937_64 gen;
    std::exponential_distribution<double> expArrival; // placeholder; lambda may change per call
    std::exponential_distribution<double> expService;

    RNG(double lambda, double mu, unsigned long long seed)
        : gen(seed), expArrival(lambda), expService(mu) {}

    // sample exponential with given rate (lambda)
    double exp_sample_rate(double rate) {
        if (rate <= 0.0) return 1e30;
        std::exponential_distribution<double> d(rate);
        return d(gen);
    }

    // sample service time (either exponential or deterministic will be handled outside)
    double nextService(double mu) {
        if (mu <= 0.0) return 1e30;
        std::exponential_distribution<double> d(mu);
        return d(gen);
    }

    // sample arrival inter arrival based on rate
    double nextArrival(double lambda) {
        if (lambda <= 0.0) return 1e30;
        std::exponential_distribution<double> d(lambda);
        return d(gen);
    }
};

/* -------------------------
DES class
------------------------- */
class DES {
public:
    Params params;
    Scenario scenario;
    State state;
    Stats stats;
    RNG rng;
    priority_queue<Event> fel;
    // arrival queue to store arrival times for delay calculations
    queue<double> arrivalTimes;

    DES(const Params& p, unsigned long long seed)
        : params(p), scenario(p.scenario), state(), stats(), rng(p.lambda, p.mu, seed)
    {
        // ensure rng seeded
    }

    void init() {
        // reset state/stats/FEL/arrivalTimes
        state = State();
        stats = Stats();
        while (!fel.empty()) fel.pop();
        while (!arrivalTimes.empty()) arrivalTimes.pop();

        // schedule initial arrival at time sampled from appropriate initial lambda
        double initialLambda = params.lambda;
        if (scenario.stepChange && state.lastEventTime >= scenario.changeTime) initialLambda = scenario.lambda2;
        double tArrival = rng.nextArrival(initialLambda);
        fel.push({tArrival, EventType::ARRIVAL});
        state.lastEventTime = 0.0;
    }

    // update time-integral statistics between lastEventTime and currentTime
    void updateTimeIntegrals(double currentTime) {
        double dt = currentTime - state.lastEventTime;
        if (dt < 0) dt = 0;
        // queue length excluding server: numInSystem - (server busy ? 1 : 0)
        int serverBusy = (state.numInSystem > 0) ? 1 : 0;
        int numInQueue = max(0, state.numInSystem - serverBusy);
        state.areaNumInQueue += numInQueue * dt;
        state.areaServerBusy += serverBusy * dt;
        state.lastEventTime = currentTime;
    }

    // returns current lambda at time t (if stepChange is enabled)
    double currentLambda(double t) const {
        if (scenario.stepChange) {
            return (t < scenario.changeTime) ? params.lambda : scenario.lambda2;
        }
        return params.lambda;
    }

    // schedule next arrival based on the current time and current lambda
    void scheduleNextArrival(double now) {
        double lam = currentLambda(now);
        double ia = rng.nextArrival(lam);
        double nextT = now + ia;
        fel.push({nextT, EventType::ARRIVAL});
    }

    void handleArrival(const Event& e) {
        double now = e.time;
        updateTimeIntegrals(now);

        // Check queue capacity:
        int capacity = scenario.finiteQueue ? scenario.queueCap : params.queueCap;
        // treat negative queueCap as infinite
        bool infinite = (capacity < 0);

        // If queue capacity finite, verify whether we can accept (numInSystem < queueCap)
        if (!infinite) {
            if (state.numInSystem >= capacity) {
                // drop
                stats.dropped++;
                // schedule next arrival and return
                scheduleNextArrival(now);
                return;
            }
        }

        // accept customer:
        state.numInSystem++;
        // push arrival time (for FIFO delay calc)
        arrivalTimes.push(now);

        // If server becomes busy (numInSystem==1), schedule departure
        if (state.numInSystem == 1) {
            double serviceTime;
            if (scenario.deterministicService || params.deterministicService) {
                serviceTime = 1.0 / params.mu;
            } else {
                serviceTime = rng.nextService(params.mu);
            }
            fel.push({now + serviceTime, EventType::DEPARTURE});
        }

        // schedule next arrival (use lambda at current time)
        scheduleNextArrival(now);
    }

    void handleDeparture(const Event& e) {
        double now = e.time;
        updateTimeIntegrals(now);

        // if no one in system, this is spurious (shouldn't happen), return
        if (state.numInSystem <= 0) return;

        // compute delay = departTime - arrivalTime of the customer served
        if (arrivalTimes.empty()) {
            // no record, but decrement count
            state.numInSystem--;
            stats.servedTotal++;
            return;
        }

        double arrivalTime = arrivalTimes.front();
        arrivalTimes.pop();

        stats.servedTotal++;
        double delay = now - arrivalTime;

        // apply warm-up: only count if we've passed warm-up threshold
        if (stats.servedTotal > params.warmup) {
            stats.servedForStats++;
            stats.totalDelay += delay;
            stats.delays.push_back(delay);
        }

        // reduce number in system
        state.numInSystem--;
        // if customers remain in system, schedule next departure
        if (state.numInSystem > 0) {
            double serviceTime;
            if (scenario.deterministicService || params.deterministicService) {
                serviceTime = 1.0 / params.mu;
            } else {
                serviceTime = rng.nextService(params.mu);
            }
            fel.push({now + serviceTime, EventType::DEPARTURE});
        }
    }

    // run until termination condition
    void run() {
        init();

        while (!fel.empty()) {
            Event e = fel.top();
            // check BY_TIME termination: if event is beyond horizonT, stop loop
            if (params.termMode == TerminationMode::BY_TIME && e.time > params.horizonT) {
                // update integrals up to horizonT for fair statistics
                updateTimeIntegrals(params.horizonT);
                stats.runTime = params.horizonT;
                break;
            }

            // check BY_SERVED termination: if we've already reached maxServed (counting total departures),
            // stop prior to processing further events. Note: warmup customers still contribute to servedTotal.
            if (params.termMode == TerminationMode::BY_SERVED && stats.servedTotal >= params.maxServed) {
                stats.runTime = state.lastEventTime;
                break;
            }

            fel.pop();

            if (e.type == EventType::ARRIVAL) handleArrival(e);
            else handleDeparture(e);
        }

        // if FEL emptied (rare), set runTime to lastEventTime
        if (stats.runTime <= 0.0) stats.runTime = state.lastEventTime;

        // if BY_SERVED mode, we may have processed an event that made servedTotal == maxServed;
        // ensure integrals are consistent (already updated within event processing)
        // set final runTime if not set
        if (stats.runTime <= 0.0) stats.runTime = state.lastEventTime;

        // finalize stats: compute any remaining integrals up to runTime already handled
        // nothing else to do
    }

    // produce metrics for this replication
    tuple<double,double,double,int,int> reportMetrics() {
        double simT = stats.runTime;
        double avgQ = (simT > 0) ? (state.areaNumInQueue / simT) : 0.0;
        double util = (simT > 0) ? (state.areaServerBusy / simT) : 0.0;
        double avgDelay = (stats.servedForStats > 0) ? (stats.totalDelay / stats.servedForStats) : 0.0;
        int served = stats.servedTotal;
        int dropped = stats.dropped;
        return {avgQ, util, avgDelay, served, dropped};
    }
};

/* -------------------------
Helper functions: CLI parser, statistics, CSV
------------------------- */
void print_usage(const char* prog) {
    cout << "Usage: " << prog << " [options]\n";
    cout << "Options:\n";
    cout << "  --lambda <double>         arrival rate lambda (default 0.9)\n";
    cout << "  --mu <double>             service rate mu (default 1.0)\n";
    cout << "  --horizonT <double>       simulation time horizon (used if --term time)\n";
    cout << "  --maxServed <int>         termination by number served (used if --term served)\n";
    cout << "  --warmup <int>            number of initial served to ignore in stats\n";
    cout << "  --term <served|time>      termination mode (default served)\n";
    cout << "  --reps <int>              number of replications (default 10)\n";
    cout << "  --seed <int>              base RNG seed (default: time-based)\n";
    cout << "  --queueCap <int>          queue capacity (default -1 infinite)\n";
    cout << "  --deterministicService    use deterministic service time = 1/mu\n";
    cout << "  --scenario_step <t> <lambda_after>\n";
    cout << "                            enable step-change in lambda at time t\n";
    cout << "  --outdir <dir>            output directory for CSV files (default .)\n";
    cout << "  --help                    show this help\n";
}

bool parseArgs(int argc, char* argv[], Params &p) {
    // simple parser: scan argv
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--lambda" && i+1 < argc) { p.lambda = stod(argv[++i]); continue; }
        if (arg == "--mu" && i+1 < argc) { p.mu = stod(argv[++i]); continue; }
        if (arg == "--horizonT" && i+1 < argc) { p.horizonT = stod(argv[++i]); continue; }
        if (arg == "--maxServed" && i+1 < argc) { p.maxServed = stoi(argv[++i]); continue; }
        if (arg == "--warmup" && i+1 < argc) { p.warmup = stoi(argv[++i]); continue; }
        if (arg == "--term" && i+1 < argc) {
            string v = argv[++i];
            if (v == "served") p.termMode = TerminationMode::BY_SERVED;
            else if (v == "time") p.termMode = TerminationMode::BY_TIME;
            else { cerr << "Unknown term mode: " << v << "\n"; return false; }
            continue;
        }
        if (arg == "--reps" && i+1 < argc) { p.reps = stoi(argv[++i]); continue; }
        if (arg == "--seed" && i+1 < argc) { p.seed = stoull(argv[++i]); continue; }
        if (arg == "--queueCap" && i+1 < argc) { p.queueCap = stoi(argv[++i]); continue; }
        if (arg == "--deterministicService") { p.deterministicService = true; continue; }
        if (arg == "--scenario_step" && i+2 < argc) {
            p.scenario.stepChange = true;
            p.scenario.changeTime = stod(argv[++i]);
            p.scenario.lambda2 = stod(argv[++i]);
            continue;
        }
        if (arg == "--outdir" && i+1 < argc) { p.outdir = argv[++i]; continue; }
        if (arg == "--help") { print_usage(argv[0]); exit(0); }
        cerr << "Unknown or incomplete option: " << arg << "\n";
        print_usage(argv[0]);
        return false;
    }
    // finalize scenario queue cap & deterministic flag
    if (p.queueCap >= 0) {
        p.scenario.finiteQueue = true;
        p.scenario.queueCap = p.queueCap;
    } else {
        p.scenario.finiteQueue = false;
        p.scenario.queueCap = -1;
    }
    if (p.deterministicService) p.scenario.deterministicService = true;
    return true;
}

// compute mean and sample standard deviation
pair<double,double> mean_and_sd(const vector<double>& v) {
    int n = (int)v.size();
    if (n == 0) return {0.0, 0.0};
    double sum = 0.0;
    for (double x : v) sum += x;
    double mean = sum / n;
    double ssum = 0.0;
    for (double x : v) ssum += (x - mean)*(x - mean);
    double sd = (n > 1) ? sqrt(ssum / (n - 1)) : 0.0;
    return {mean, sd};
}

// write CSV helper
void write_csv_per_rep(const string& path, const vector<tuple<int,double,double,double,int,int>>& rows) {
    // rows: (repIndex, avgQ, util, avgDelay, served, dropped)
    ofstream f(path);
    f << "rep,avgQ,util,avgDelay,served,dropped\n";
    for (const auto& r : rows) {
        int rep; double avgQ, util, avgDelay; int served, dropped;
        tie(rep, avgQ, util, avgDelay, served, dropped) = r;
        f << rep << "," << setprecision(10) << avgQ << "," << util << "," << avgDelay << "," << served << "," << dropped << "\n";
    }
    f.close();
}

void write_csv_summary(const string& path,
    double mean_avgQ, double ci_avgQ,
    double mean_util, double ci_util,
    double mean_delay, double ci_delay,
    int reps)
{
    ofstream f(path);
    f << "metric,mean,ci95_lower,ci95_upper,n\n";
    f << "avgQ," << mean_avgQ << "," << (mean_avgQ - ci_avgQ) << "," << (mean_avgQ + ci_avgQ) << "," << reps << "\n";
    f << "util," << mean_util << "," << (mean_util - ci_util) << "," << (mean_util + ci_util) << "," << reps << "\n";
    f << "avgDelay," << mean_delay << "," << (mean_delay - ci_delay) << "," << (mean_delay + ci_delay) << "," << reps << "\n";
    f.close();
}

/* -------------------------
Main entry: run multiple replications, compute CI, write CSV
------------------------- */
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Params p;
    // default seed: use time
    p.seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() & 0xffffffffULL;
    if (!parseArgs(argc, argv, p)) return 1;

    cout << "DES M/M/1 simulation\n";
    cout << "lambda=" << p.lambda << " mu=" << p.mu << " termMode=" << ((p.termMode==TerminationMode::BY_SERVED)?"served":"time") << "\n";
    cout << "reps=" << p.reps << " warmup=" << p.warmup << " queueCap=" << p.queueCap << " deterministicService=" << (p.deterministicService? "true":"false") << "\n";
    if (p.scenario.stepChange) {
        cout << "Scenario: step-change lambda at t=" << p.scenario.changeTime << " -> lambda=" << p.scenario.lambda2 << "\n";
    }

    // storage per replication
    vector<tuple<int,double,double,double,int,int>> perRep; // repIndex, avgQ, util, avgDelay, served, dropped
    vector<double> avgQs, utils, avgDelays;

    // run reps
    for (int rep = 0; rep < p.reps; ++rep) {
        unsigned long long seed = p.seed + rep + 1ULL; // ensure different seeds
        DES sim(p, seed);

        sim.run();

        double avgQ, util, avgDelay;
        int served, dropped;
        tie(avgQ, util, avgDelay, served, dropped) = sim.reportMetrics();

        cout << fixed << setprecision(6);
        cout << "rep " << rep+1 << ": avgQ=" << avgQ << " util=" << util << " avgDelay=" << avgDelay<< " servedTotal=" << served << " dropped=" << dropped << "\n";

        perRep.emplace_back(rep+1, avgQ, util, avgDelay, served, dropped);
        avgQs.push_back(avgQ);
        utils.push_back(util);
        avgDelays.push_back(avgDelay);
    }

    // compute mean and CI95 (1.96 * s / sqrt(n))
    auto [meanQ, sdQ] = mean_and_sd(avgQs);
    auto [meanU, sdU] = mean_and_sd(utils);
    auto [meanD, sdD] = mean_and_sd(avgDelays);
    int n = (int)avgQs.size();
    double ciQ = (n>0) ? (1.96 * sdQ / sqrt(n)) : 0.0;
    double ciU = (n>0) ? (1.96 * sdU / sqrt(n)) : 0.0;
    double ciD = (n>0) ? (1.96 * sdD / sqrt(n)) : 0.0;

    // write CSVs
    string out_per = p.outdir + "/results_per_rep.csv";
    string out_sum = p.outdir + "/summary.csv";
    write_csv_per_rep(out_per, perRep);
    write_csv_summary(out_sum, meanQ, ciQ, meanU, ciU, meanD, ciD, n);

    cout << "\nSummary across " << n << " reps:\n";
    cout << "avgQ mean=" << meanQ << " ci95=" << ciQ << "\n";
    cout << "util mean=" << meanU << " ci95=" << ciU << "\n";
    cout << "avgDelay mean=" << meanD << " ci95=" << ciD << "\n";
    cout << "Wrote: " << out_per << " and " << out_sum << "\n";

    return 0;
}
