#include "Enclave.h"
#include "Enclave_t.h"
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "sgx_trts.h"
#include <limits>
#include <iostream>
#define  _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

using namespace std;

int indexs[5+5]={1,2,0,3,1,4}; //待配置
int insert = 4;
int Ne = 5;
int Nt = 6;

struct Element {
    float value;
    int index;
};

void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap); 
    ocall_print_string(buf);
}

float gaussrand_NORMAL() {
    float V1, V2, S;
        float X;
                do {
            char a[10];
            char b[10];
            sgx_read_rand(reinterpret_cast<unsigned char*>(&a),sizeof(a));
            sgx_read_rand(reinterpret_cast<unsigned char*>(&b),sizeof(b));
                        float U1 = static_cast<float>(*(unsigned char *)a)/256.0f;
                        float U2 = static_cast<float>(*(unsigned char *)b)/256.0f;
                        V1 = 2 * U1 - 1;
                        V2 = 2 * U2 - 1;
                        S = V1 * V1 + V2 * V2;
                } while (S >= 1 || S <= 1e-6);
                X = V1 * sqrtf(-2 * logf(S+1e-11f) / S);
        return X;
}

float gaussrand(float mean, float stdc) {
        return mean + gaussrand_NORMAL() * stdc;
}

double sgx_arr[10][128] = {-0.413019, -0.648176, -0.738965, -0.94625, -0.787735, 0.162664, 0.855689, 0.296127, 0.022478, 0.443469, 0.942409, -0.283812, 0.068085, -0.829333, 0.624474, -0.611512, -0.840649, -0.801917, 0.196708, 0.167713, -0.769749, -0.917539, -0.831678, 0.493822, 0.980103, 0.909316, -0.315548, -0.783482, 0.153313, -0.177654, -0.661152, -0.636425, 0.813227, -0.253926, 0.913525, 0.759812, -0.731122, -0.378765, 0.306895, -0.996104, -0.188659, 0.734144, 0.43832, -0.907707, -0.121548, -0.576072, 0.991638, -0.555703, 0.258666, 0.896797, -0.18211, 0.085026, -0.990219, -0.417657, 0.335172, 0.353607, 0.730925, -0.433116, 0.131238, 0.457873, 0.851551, 0.242515, 0.43129, 0.669351, -0.939598, 0.790791, 0.07347, -0.644559, -0.884442, -0.741217, -0.508293, 0.249204, -0.308592, 0.571099, 0.820891, 0.377655, 0.676666, -0.884371, -0.002113, 0.890487, -0.082762, 0.786785, 0.168079, -0.256825, 0.291531, 0.1631, 0.504951, -0.946527, 0.02267, 0.748698, -0.376927, 0.630698, 0.301554, -0.492495, 0.644448, 0.732006, 0.185314, -0.895886, -0.097512, -0.998885, 0.767601, 0.724479, 0.355212, 0.357027, 0.421804, 0.430587, -0.574356, 0.577541, -0.142626, -0.494353, 0.651517, -0.145511, 0.967618, 0.185087, -0.425051, -0.044654, -0.086515, -0.615955, 0.979642, -0.644917, -0.232692, -0.265945, -0.430869, -0.556808, 0.108713, 0.245, 0.803278, -0.754665,
0.420703, -0.529539, 0.495861, -0.86936, -0.869864, -0.916696, 0.521018, 0.299828, -0.043925, -0.314319, -0.447982, -0.173597, 0.345998, -0.474339, -0.712097, 0.415244, 0.270195, -0.458431, 0.817267, 0.389846, 0.203676, 0.930971, 0.096953, -0.280005, 0.135916, 0.186633, -0.315872, 0.388978, 0.691847, -0.127426, 0.65889, 0.138309, 0.207046, -0.740637, -0.187584, 0.937982, -0.774958, 0.078726, -0.548511, 0.670203, 0.221834, -0.255117, -0.591096, 0.434223, 0.266535, 0.791996, 0.735776, -0.886526, -0.928048, -0.528174, -0.811105, 0.368926, 0.522331, 0.550746, -0.700383, -0.06747, -0.737163, -0.120161, 0.910405, -0.004433, 0.743641, -0.361578, -0.527923, -0.788883, 0.604422, 0.381966, 0.59205, 0.359441, -0.064645, 0.22181, -0.665297, -0.917389, 0.763778, 0.941509, -0.002481, -0.378019, -0.536694, 0.819572, 0.254516, 0.762784, 0.124234, 0.586586, 0.943696, 0.710183, -0.885489, -0.098143, -0.292981, 0.43312, 0.906848, -0.230391, -0.185604, 0.049871, 0.945854, -0.224995, -0.992853, 0.31274, -0.838917, -0.69046, 0.215614, -0.253144, 0.989716, -0.492364, -0.872724, 0.024907, -0.827176, 0.458509, 0.757806, 0.547988, 0.698675, -0.846329, 0.070607, -0.126691, -0.239857, 0.582154, 0.44086, -0.03253, -0.374366, 0.100211, 0.073514, -0.922076, 0.214262, 0.680087, -0.708626, -0.377121, -0.178296, 0.813442, -0.949186, 0.156337,
0.755409, -0.515239, -0.08725, -0.880813, 0.485309, -0.158034, -0.295507, -0.633229, -0.921783, -0.455953, 0.469826, -0.789874, 0.272338, -0.328709, 0.376298, -0.558712, 0.983358, 0.395939, 0.802374, -0.092426, -0.605635, 0.198481, 0.129731, 0.437877, 0.100955, 0.936058, 0.074651, -0.947808, -0.913774, 0.470292, 0.463406, 0.840859, 0.420283, 0.16102, -0.712614, -0.13648, -0.317881, 0.741116, -0.178559, 0.882601, -0.607016, -0.722486, 0.405318, 0.839045, -0.089046, -0.431146, 0.258575, 0.032194, -0.756331, -0.661926, -0.012208, -0.347155, 0.38354, -0.826605, -0.022558, 0.143189, -0.78085, 0.344824, 0.416162, -0.226674, 0.403987, -0.700357, -0.837391, -0.092618, -0.642095, 0.803979, 0.351008, -0.901716, -0.961159, -0.074409, -0.332478, 0.08398, 0.764908, -0.948897, -0.163105, 0.02324, -0.273622, 0.650001, -0.352549, 0.422852, 0.765192, -0.816161, -0.755776, 0.461972, 0.64856, 0.974855, 0.544158, -0.506941, -0.802904, -0.179063, -0.712, 0.471701, 0.584697, -0.859375, 0.112067, -0.3369, 0.227253, -0.412467, 0.035761, -0.606878, 0.753269, 0.455096, -0.696048, 0.060389, 0.298783, 0.887394, 0.759367, -0.013259, -0.633346, -0.021321, -0.218632, 0.837178, 0.961425, 0.259666, 0.821678, -0.171806, -0.981921, 0.643795, 0.657434, 0.946345, -0.047698, -0.880307, 0.081543, -0.350908, -0.430486, -0.247403, 0.306219, 0.064122,
0.042188, -0.807475, -0.177602, 0.523249, -0.767931, -0.89836, -0.437781, 0.295426, 0.945815, 0.352285, -0.062027, -0.830847, -0.77131, -0.416227, 0.31883, 0.604325, -0.75691, -0.967632, 0.236233, -0.895317, -0.28399, -0.322514, -0.355278, -0.912827, -0.119137, -0.314143, 0.950869, -0.359748, -0.408218, -0.404256, 0.766263, 0.567001, -0.054733, -0.107669, 0.407604, 0.586187, 0.728401, 0.628732, 0.598416, 0.014242, -0.548322, 0.154004, -0.594262, 0.526201, -0.165572, -0.988974, -0.710309, 0.625761, 0.514455, -0.961507, -0.396917, -0.678286, 0.845309, -0.151426, -0.331387, 0.271744, -0.874375, -0.486473, -0.558596, -0.738446, -0.799694, -0.575576, 0.515072, -0.935007, -0.203502, -0.701085, -0.313475, 0.443124, -0.608373, 0.223724, 0.228415, -0.798195, -0.87171, 0.795508, 0.176813, 0.135996, -0.115881, 0.827591, 0.984357, 0.063111, -0.462057, -0.483238, -0.346364, -0.921811, 0.400522, 0.043363, -0.975391, -0.402485, -0.655402, 0.119121, -0.689315, 0.426045, -0.363352, 0.145615, 0.726284, -0.505802, 0.512222, 0.32779, 0.254555, 0.328934, -0.555304, -0.10854, -0.094574, 0.979167, -0.858489, -0.92748, -0.344415, 0.432965, 0.697412, -0.126592, 0.971792, -0.146516, -0.263641, 0.501158, 0.363046, 0.785152, -0.40698, -0.353177, -0.689739, -0.477233, -0.183677, 0.975008, -0.086192, 0.202973, -0.935534, -0.424362, -0.523241, 0.509286,
0.761497, -0.460531, 0.882477, 0.971827, -0.469504, 0.128834, 0.564343, 0.811005, -0.347694, -0.595351, 0.339551, 0.602279, -0.57808, -0.927593, -0.131904, 0.512646, 0.166297, 0.421582, 0.711664, 0.966827, -0.285135, -0.974716, -0.008559, -0.860488, 0.40766, 0.310587, 0.630566, 0.940362, 0.000982, 0.86001, -0.056546, 0.782747, 0.34985, -0.622217, 0.467961, -0.418831, 0.784801, 0.221277, 0.058106, 0.314996, -0.81486, 0.818566, -0.082165, 0.103209, -0.925699, 0.054892, 0.597515, 0.504189, 0.317693, 0.527206, 0.850261, 0.506342, -0.50567, 0.808897, -0.164603, -0.638184, 0.278484, -0.964481, -0.736607, 0.232914, -0.983187, -0.857499, 0.379896, -0.243903, -0.134283, -0.651809, 0.892695, -0.825777, -0.500703, 0.462395, 0.194608, 0.143005, 0.544328, 0.976007, -0.938974, -0.003817, 0.477705, 0.538325, -0.337273, -0.163812, -0.425323, -0.481101, 0.909652, 0.783331, -0.254997, -0.926639, 0.787799, -0.707885, -0.506423, 0.070427, 0.350171, 0.483099, 0.874701, 0.126263, -0.368947, -0.836887, -0.386669, 0.367989, -0.970118, -0.834889, -0.496922, -0.746685, 0.747088, 0.754248, 0.415934, 0.205544, 0.402041, -0.79578, 0.193091, 0.055527, 0.045365, -0.88067, 0.503695, 0.048921, 0.788112, 0.297014, -0.357256, -0.018185, 0.5254, 0.349831, 0.949016, 0.574525, -0.109432, 0.064325, 0.427896, 0.320158, 0.737926, -0.541026,
-0.416605, -0.890534, -0.550187, 0.008184, -0.23658, 0.883139, 0.462428, -0.491309, -0.487804, -0.55406, -0.613574, -0.780442, -0.72655, 0.663574, -0.433668, 0.634202, -0.830955, -0.13873, 0.441097, 0.462732, 0.859874, 0.22881, 0.82955, -0.523095, -0.612135, -0.77466, -0.237048, 0.572229, -0.093376, -0.418472, 0.172205, 0.671122, 0.957563, -0.481683, -0.204678, -0.763283, -0.417181, 0.649233, 0.342237, -0.489968, 0.89799, -0.689071, -0.063836, 0.542778, -0.303386, -0.538764, 0.921952, -0.526371, -0.270016, -0.459484, -0.000498, 0.002924, -0.398012, -0.776191, 0.462282, 0.005146, 0.22812, 0.397954, -0.753635, -0.604073, 0.768659, -0.950577, -0.834129, 0.053027, -0.885724, -0.357284, -0.293828, -0.556781, 0.609935, 0.547785, -0.216324, -0.18512, -0.181555, -0.771723, -0.271605, -0.694106, -0.603647, 0.023844, -0.715657, 0.007379, -0.2763, 0.363063, 0.244926, 0.100117, 0.970604, -0.352787, 0.201846, -0.002992, 0.137425, 0.013021, 0.17798, 0.842025, 0.982712, -0.772346, -0.083114, 0.847313, 0.549433, 0.619293, -0.840565, 0.161361, 0.485664, 0.012196, -0.40446, -0.589093, 0.612568, -0.174835, -0.987479, 0.058472, 0.397528, 0.316669, 0.426237, 0.299619, -0.821065, -0.872495, -0.899045, -0.772628, 0.398385, -0.387737, -0.056735, 0.014936, 0.880422, -0.768532, 0.7186, -0.458492, -0.594428, 0.455708, 0.760663, 0.214351,
-0.302103, 0.361906, 0.181962, 0.037618, 0.947802, -0.859493, 0.194569, -0.764173, 0.682767, -0.787726, -0.314927, 0.531181, 0.147666, 0.371491, -0.799901, 0.571506, 0.324251, 0.02605, 0.382461, 0.162908, 0.639308, -0.762674, -0.055016, -0.813937, 0.245778, 0.230073, 0.559919, 0.371428, -0.53176, -0.548415, 0.530644, 0.713553, -0.442932, 0.871614, 0.389535, -0.511606, -0.678027, 0.490272, 0.055269, -0.054723, -0.024468, -0.888119, -0.600733, -0.903646, -0.083257, -0.831285, 0.964124, -0.106913, -0.640422, -0.461294, 0.954761, -0.641971, 0.314054, -0.906892, -0.122912, 0.706228, -0.526711, -0.734298, -0.167282, 0.141931, 0.719613, 0.555283, -0.782741, -0.796031, 0.532645, -0.92928, 0.153484, -0.549746, 0.095209, 0.399009, 0.477826, 0.079318, -0.57636, 0.273455, -0.1407, -0.888822, 0.46679, 0.232506, -0.016926, 0.994063, -0.052822, 0.865, 0.147453, 0.578119, 0.60022, 0.867796, 0.270862, 0.916996, 0.294354, 0.837203, -0.974753, -0.561012, 0.408967, 0.513267, -0.495498, -0.984852, 0.170994, 0.858864, -0.754518, -0.298817, 0.716245, -0.948216, 0.23396, 0.059492, 0.390835, -0.310483, -0.030846, 0.308502, -0.072712, 0.118196, 0.055686, -0.411919, -0.182943, -0.06982, -0.029518, 0.691745, 0.33913, 0.215339, -0.787881, -0.72638, 0.828163, 0.769841, 0.687637, 0.76546, -0.293439, 0.331303, 0.49068, 0.560516,
-0.935059, -0.328568, -0.584971, -0.032822, -0.370481, 0.265033, -0.214596, -0.039402, 0.281615, -0.825284, -0.45881, -0.45234, -0.136646, -0.683798, 0.244127, 0.472372, 0.510535, -0.1186, 0.921412, -0.309698, -0.804737, -0.196266, -0.683643, 0.929001, -0.637775, -0.568353, 0.552412, -0.70914, 0.66883, 0.948789, -0.239929, -0.293338, -0.251053, -0.570116, 0.957889, -0.784427, -0.310961, -0.95003, -0.537288, -0.948124, 0.606097, 0.723066, 0.499993, 0.86204, -0.53475, -0.891929, -0.646039, 0.280501, -0.141357, 0.401215, 0.587331, -0.493284, -0.232589, -0.275398, -0.11578, -0.386519, -0.338137, 0.511279, -0.651563, -0.666812, -0.847342, -0.755315, 0.794408, -0.038528, 0.031073, -0.920878, 0.064591, -0.484781, 0.516784, 0.649439, 0.910037, -0.015486, 0.029257, 0.364273, 0.12379, 0.531152, 0.752064, 0.701619, 0.112365, 0.582919, -0.17849, 0.487171, -0.82764, -0.194342, -0.49969, 0.862015, 0.257089, 0.786369, 0.739453, -0.615402, 0.680496, -0.996948, 0.146602, 0.515195, -0.05972, -0.245515, 0.005755, -0.274805, -0.86227, 0.304697, 0.397752, -0.641623, 0.59621, -0.683161, -0.022293, -0.384739, 0.004468, 0.080905, -0.563185, -0.510188, -0.936161, -0.597394, 0.512548, 0.629422, 0.196769, -0.564594, 0.556217, 0.024171, -0.862254, -0.726366, -0.677007, 0.92548, -0.671521, 0.324161, 0.738961, -0.431469, -0.495578, -0.304355,
0.606447, 0.556742, -0.240821, 0.865194, -0.153572, -0.184342, -0.248215, -0.560151, -0.536688, 0.779366, -0.423256, 0.432959, 0.672193, -0.828961, -0.135953, -0.1893, 0.160636, -0.707777, 0.832428, 0.866381, 0.870434, -0.382932, -0.458625, -0.883539, -0.14547, -0.760476, -0.473085, -0.752699, -0.838545, -0.147052, 0.272786, -0.27381, 0.46496, 0.034894, 0.425543, -0.270141, 0.447065, 0.346384, 0.397642, 0.759522, 0.833067, 0.98394, -0.300024, -0.864666, 0.509733, 0.005214, -0.818177, -0.829412, 0.636227, -0.067076, -0.076658, -0.219166, 0.239989, -0.287722, 0.993687, 0.222394, -0.494354, -0.268949, -0.854196, 0.093126, -0.203124, 0.622618, 0.894541, -0.936235, -0.013312, -0.046574, 0.201501, -0.856741, -0.865599, 0.85178, -0.345884, 0.942905, 0.089691, -0.647273, -0.427117, -0.136798, 0.220177, 0.192693, -0.722519, 0.098841, -0.402538, 0.519375, -0.241148, -0.093233, -0.30397, -0.674681, 0.831559, 0.042902, 0.205898, -0.919567, -0.870211, -0.853746, 0.17048, 0.705611, -0.604431, -0.581845, 0.839958, 0.224232, 0.550031, 0.240933, -0.072647, 0.298294, -0.895265, -0.291364, 0.152239, -0.318782, 0.583945, -0.147703, -0.602929, -0.38663, -0.431528, -0.727271, 0.318214, 0.728549, -0.633372, 0.155073, 0.105499, 0.501657, -0.40483, -0.250473, 0.896167, -0.929652, 0.015642, 0.141283, -0.355291, -0.95562, -0.734827, 0.454175,
-0.674519, -0.536586, 0.580108, 0.299049, -0.546081, -0.452153, -0.327862, 0.195435, 0.396067, 0.874939, 0.93605, -0.671107, 0.309983, -0.317469, -0.143525, 0.502216, 0.872154, -0.850347, 0.173905, 0.125311, 0.707519, 0.835967, 0.795166, 0.519291, 0.076401, 0.039859, 0.587105, 0.162785, -0.47173, -0.792043, -0.095712, 0.652227, -0.005201, -0.851948, -0.433994, 0.159268, -0.864806, -0.969362, 0.183608, 0.285788, -0.709921, 0.4634, -0.213325, -0.083225, 0.092181, 0.426753, -0.055166, 0.555573, -0.275772, -0.624886, 0.542301, 0.599411, 0.138395, 0.896781, -0.614716, 0.232178, 0.938641, -0.70204, 0.597084, 0.907337, -0.416796, -0.790011, -0.009691, 0.885905, 0.75775, -0.886314, 0.253083, 0.183328, 0.923916, -0.465909, 0.429538, -0.182296, 0.919992, -0.585444, -0.788225, 0.751125, 0.684207, 0.101696, -0.941137, 0.252254, -0.491351, -0.596157, 0.258878, 0.921679, 0.940105, 0.4049, 0.089472, 0.114974, -0.394493, -0.773012, -0.825772, 0.166343, 0.484613, 0.478587, -0.466759, -0.145944, 0.177703, 0.265973, 0.149725, -0.557244, -0.103823, -0.931036, -0.302548, -0.865546, 0.041901, 0.655376, 0.959052, 0.917402, -0.937276, 0.610186, -0.82464, -0.911334, -0.43278, -0.581388, 0.569004, -0.332484, 0.968195, -0.509025, -0.052482, 0.496285, 0.836222, 0.025237, 0.722781, 0.561562, 0.035722, -0.796505, -0.78503, 0.497707};

void emb_en(uint32_t dim, float* values,float range){
    for (int i = 0; i < dim-1; i++) {
        unsigned char tmp;
        float X1 = static_cast<float>(sgx_read_rand(reinterpret_cast<unsigned char*>(&tmp), sizeof(tmp)))/static_cast<float>(256);
        values[i] = (2*X1-1)*range;
    }
    values[dim-1] = 0;
    for (int i = 0; i < dim-1; i++) {
        values[dim-1] += values[i];
    }
    int k = static_cast<int>(fabs(values[dim-1] * 100))%10;
    for (int i = 0; i < dim-1; i++) {
        values[i] = values[i] - sgx_arr[k][i%128];
    }
    return;
}

void ecall_ceshi(float* input,float* output,int M,int C){
    for (int i = 0; i < M; i++) {
        int k = static_cast<int>(fabs(input[i*C+C-1] * 100))%10;
        for (int j = 0; j < C-1; j++) {
            output[i*(C-1)+j] = input[i*C+j] + sgx_arr[k][j%128];
    }
    }
}
void ecall_ceshi_grad(float* input,float* output,int M,int C){
    for (int i = 0; i < M; i++){
        output[i*C+C-1] = 0.0f;
        for (int j = 0; j < C-1; j++) {
            output[i*C+j] = input[i*(C-1)+j];
        }
    }
}

void emb_ss(float* input,float* output,int M,int C,int num,int dim){
    int N = M*num*dim;
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < num; j++) {
            int idx = static_cast<int>(fabs(input[i*C+j*(dim+1)+dim] * 100))%10;
            for (int k = 0; k < dim; k++) {
                x[i*num*dim+j*dim+k] = input[i*C+j*(dim+1)+k] + sgx_arr[idx][k%128];
            }
            }
        }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    return;
}

void emb_ss_grad(float* grad,float* output,int M,int C,int num,int dim){
    int N = M*num*dim; 
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < num; j++) {
            output[i*C+j*(dim+1)+dim] = 0.0f;
            for (int k = 0; k < dim; k++) {
                output[i*C+j*(dim+1)+k] = x[i*num*dim+j*dim+k];
            }
            }
        }
    free(x);
    return;
}

void ecall_encrypt(float *input, int N, float *output){
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (input[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (input[j]*input[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(input[i % N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = input[j-N*(Ne-1)];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] = sum;
    }
    return;
}

void ecall_decrypt(float *input,int N, float *output){
    for(int j=0;j<N;j++){
        output[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] = sum;
    }
    return;
}

void ecall_relu(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
		
    for (int j = 0; j < N; j++) {
        x[j]= x[j] >= 1e-7f ? x[j] : 0.0f;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
	if(output[j]<1e-6 && output[j]>-1e-6){
            output[j]= 0.0f;
        }  
    }
    free(x);
    return;
    }


void ecall_sigmoid(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    for (int j = 0; j < N; j++) {
        x[j]= static_cast<float>(1)/(static_cast<float>(1) + expf(-x[j]));
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    return;
    }

void ecall_logloss(float *input,int N,int M,int C,float *label,float *weight,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    output[0] = 0.0f;
    float epsilon = 1e-7f;
    for (int i = 0; i < M; i++ ){
            for (int j = 0; j< C;j++){
            output[0] = output[0] - ((label[i*C+j] * logf(x[i*C+j] + epsilon) ) + (1 - label[i*C+j]) * logf(1 - x[i*C+j] + epsilon))* weight[i];
            }
    }
    free(x);
    return;
}

void ecall_relu_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;  
    }

    for (int j = 0; j < N; j++) {
        g[j]= x[j] >= 1e-7f ? g[j] : 0.0f;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
	if(output[j]<1e-6 && output[j]>-1e-6){
            output[j]= 0.0f;
        }  
    }
    free(x);
    free(g);
    return;
    }

void ecall_sigmoid_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        g[j]= g[j]*((static_cast<float>(1) / (static_cast<float>(1) + expf(-x[j])))*(static_cast<float>(1)-(static_cast<float>(1) / (static_cast<float>(1) + expf(-x[j])))));
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
    }

void ecall_logloss_grad(float *input,float grad,int N,int M,int C,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    float epsilon = 1e-7f;

    for (int i = 0; i <M; i++ ){
        for (int j = 0; j< C;j++){
        grad_l[i*C+j] = weight[i]*grad * (logf(1 - x[i*C+j] + epsilon)-logf(x[i*C+j] + epsilon));
        }
    }

    for (int i = 0; i <M; i++ ){
        grad_w[i] = 0;
        for (int j = 0; j< C;j++){
            grad_w[i] -= grad*((label[i*C+j] * logf(x[i*C+j] + epsilon)) + (1 - label[i*C+j]) * logf(1 - x[i*C+j] + epsilon));
        }
    }

    for (int i = 0; i <M; i++ ){
            for (int j = 0; j< C;j++){
            x[i*C+j] = weight[i]*grad * (x[i*C+j]- label[i*C+j] + epsilon *(1-2*label[i*C+j]))/((x[i*C+j]+epsilon)*(1-x[i*C+j]+epsilon));
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_x[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_x[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * grad_x[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        grad_x[j] =x[j-(Ne-1)*N] + sum;
    }
    free(x);
    return;
    }

void ecall_tanh(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float exp_pos = expf(x[j]);
        float exp_neg = expf(-x[j]);
        if(exp_pos == INFINITY || exp_neg == INFINITY){
            x[j] = x[j] > 0 ? 1.0f : -1.0f;
        }
        else{
            x[j]= (exp_pos-exp_neg)/(exp_pos+exp_neg);
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    return;
}

void ecall_tanh_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        float exp_pos = expf(x[j]);
        float exp_neg = expf(-x[j]);
        float tmp;
        if(exp_pos == INFINITY || exp_neg == INFINITY){
            tmp = 0.0f;
        }
        else{
            tmp = static_cast<float>(4)/((exp_pos+exp_neg)*(exp_pos+exp_neg));
        }
        g[j] = g[j]*tmp;
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_softplus(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    
    for (int j = 0; j < N; j++) {
        if(x[j] > 50.0f){
            x[j] = x[j];
        }
        else if(x[j] < -50.0f){
            x[j] = 0.0f;
        }
        else{
            x[j]= logf(static_cast<float>(1)+expf(x[j]));
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
}

void ecall_softplus_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float tmp;
        if(x[j] > 50.0f){
            tmp = 1.0f;
        }
        else if(x[j] < -50.0f){
            tmp = 0.0f;
        }
        else{
            tmp = static_cast<float>(1)/(static_cast<float>(1) + expf(-x[j]));
        }
        g[j] = g[j]*tmp;
    }
    
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_leakyrelu(float *input,int N,float *output,float alpha){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    for (int j = 0; j < N; j++) {
        x[j]= x[j] >= 1e-7f ? x[j] : alpha*x[j];
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_leakyrelu_grad(float *input,float *grad,int N,float *output,float alpha){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        g[j] = x[j] >= 1e-7 ? g[j] : alpha * g[j];
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_abs(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int j = 0; j < N; j++) {
        if(fabs(x[j]) < 1e-7f){
            x[j] = 0.0f;
        }
        else if(x[j] >= 1e-7f){
            x[j] = x[j];
        }
        else{
            x[j] = -1.0f * x[j];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_abs_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float tmp;
        if(fabs(x[j]) < 1e-7f){
            tmp = 0.0f;
        }
        else if(x[j] >= 1e-7f){
            tmp = 1.0f;
        }
        else{
            tmp = -1.0f;
        }
        g[j] = g[j]*tmp;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_log(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int j = 0; j < N; j++) {
        x[j] = logf(x[j]+10e-4);
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_log_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float tmp = static_cast<float>(1)/(x[j]+10e-4f);
        g[j] = g[j]*tmp;
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_exp(float *input, int N, float *output) {
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int j = 0; j < N; j++) {
        x[j] = expf(x[j]);
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
}

void ecall_exp_grad(float *input, float *grad, int N, float *output) {
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float tmp = expf(x[j]);
        g[j] = g[j] * tmp;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_sign(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int j = 0; j < N; j++) {
        if(fabs(x[j]) < 1e-7f){
            x[j] = 0.0f;
        }
        else if(x[j] >= 1e-7f){
            x[j] = 1.0f;
        }
        else{
            x[j] = -1.0f;
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_sign_grad(float *input,float *grad,int N,float *output){
    for(int j=0;j<N*Ne;j++){
        output[j] = 0.0f;
    }
    return;
}

void ecall_square(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        x[j] = x[j] * x[j];
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_square_grad(float *input,float *grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        float tmp = 2.0f*x[j];
        g[j] = g[j]*tmp;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    return;
}

void ecall_lessequal(float *input,int N,float *output,float alpha){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        x[j]= (x[j] - alpha) <= 1e-7f ? 1.0f : 0.0f;
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_greater(float *input,int N,float *output,float alpha){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        x[j]= x[j] > alpha ? 1.0f : 0.0f;
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_reducesum(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    output[0] = 0.0f;
    for (int j = 0; j < N; j++) {
        output[0] += x[j];
    }
    return;
    }

void ecall_reducesum_grad(float *input,float grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for (int j = 0; j < N; j++) {
        x[j] = grad;
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
}

void ecall_reducemean(float *input,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    output[0] =0.0f;
    for (int j = 0; j < N; j++){
        output[0] += x[j]/static_cast<float>(N);
    }
    return;
    }

void ecall_reducemean_grad(float *input,float grad,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for (int j = 0; j < N; j++) {
        x[j] = grad/static_cast<float>(N);
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
}

void ecall_maximum(float *input1,float *input2,int N,float *output){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }

    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        x1[j] = x1[j] >= x2[j] ? x1[j] : x2[j];
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N*(Ne-1)] + sum; 
    }
    free(x1);
    free(x2);
    return;
    }

void ecall_maximum_grad(float *input1,float *input2,float *grad,int N,float *output1,float *output2){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        if(x1[j] >= x2[j]){
            x1[j] = g[j];
            x2[j] = 0.0f;
        }
        else{
            x1[j] = 0.0f;
            x2[j] = g[j];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x2[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output2[i] =(x2[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x2);
    free(g);
    return;
}

void ecall_minimum(float *input1,float *input2,int N,float *output){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }

    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }
    for (int j = 0; j < N; j++) {
        x1[j] = x1[j] <= x2[j] ? x1[j] : x2[j];
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N*(Ne-1)] + sum; 
    }
    free(x1);
    free(x2);
    return;
    }

void ecall_minimum_grad(float *input1,float *input2,float *grad,int N,float *output1,float *output2){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int j = 0; j < N; j++) {
        if(x1[j] <= x2[j]){
            x1[j] = g[j];
            x2[j] = 0.0f;
        }
        else{
            x1[j] = 0.0f;
            x2[j] = g[j];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N*(Ne-1)] + sum; 
    }
    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x2[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output2[i] =(x2[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x2);
    free(g);
    return;
}

void ecall_softmax(float *input,int N,int M,int C,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int i = 0; i < M; i++) {
            float max_val = x[i*C];
            for (int j = 1; j < C; j++) {
                max_val = fmaxf(max_val, x[i*C+j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < C; j++) {
                sum += expf(x[i*C+j] - max_val);
            }
            for (int j = 0; j < C; j++) {
                x[i*C+j] = expf(x[i*C+j] - max_val) / sum;
            }
        }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
    }

void ecall_softmax_grad(float *input, float *grad, int N, int M, int C, float *output) {
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    for (int i = 0; i < M; i++) {
        float max_val = x[i * C];
        for (int j = 1; j < C; j++) {
            max_val = fmaxf(max_val, x[i * C + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float scaled_input = x[i * C + j] - max_val;
            sum += expf(scaled_input);
        }
        for (int j = 0; j < C; j++) {
            float scaled_input = x[i * C + j] - max_val;
            x[i * C + j] = expf(scaled_input) / sum;
        }
    }
    float *gt = (float *)malloc(sizeof(float) * M);
    memset(gt, 0, sizeof(float) * M);
    for (int i = 0; i < M; i++) {
        gt[i] = 0.0f;
        for (int j = 0; j < C; j++) {
            gt[i] += g[i * C + j] * x[i * C + j];
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            g[i*C+j] = (g[i * C + j] - gt[i]) * x[i * C + j];
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(g);
    free(x);
    free(gt);
    return;
}

void ecall_clip(float *input, float* down, float* up,int n1,int n2,int N,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    
    float *down1 = (float*)malloc(sizeof(float)*n1);
    memset(down1, 0, sizeof(float)*n1);
    if(n1 == 1){
        down1[0] = down[0];
    }
    else{
        for(int j=0;j<n1;j++){
        down1[j] = 0.0f;
        float sum = down[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = down[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        down1[j] = sum;
    }
    }    
 
    float *up1 = (float*)malloc(sizeof(float)*n2);
        memset(up1, 0, sizeof(float)*n2);
    if(n2 == 1){
        up1[0] = up[0];
    }
    else{
        for(int j=0;j<N;j++){
        up1[j] = 0.0f;
        float sum = up[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = up[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        up1[j] = sum;
    }
    }
    for (int j = 0; j < N; j++) {
        if(x[j]-down1[j%n1] < -1e-7f){
            x[j] = down1[j%n1];
            //x[j] = 1.0f;
        }
        if(x[j]-up1[j%n2] > 1e-7f){
            x[j] = up1[j%n2];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    free(up1);
    free(down1);
    return;
}

void ecall_clip_grad(float *input,float *grad, float* down, float* up,int n1,int n2,int N,float *output){
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    float *up1 = (float*)malloc(sizeof(float)*n2);
        memset(up1, 0, sizeof(float)*n2);
    if(n2 == 1){
        for(int j=0;j<n2;j++){
            up1[j] = up[j];
        }
    }
    else{
        for(int j=0;j<n2;j++){
        up1[j] = 0.0f;
        float sum = up[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = up[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        up1[j] = sum;
    }
    }
    float *down1 = (float*)malloc(sizeof(float)*n1);
        memset(down1, 0, sizeof(float)*n1);
    if(n1 == 1){
        for(int j=0;j<n1;j++){
            down1[j] = down[j];
        }
    }
    else{
        for(int j=0;j<n1;j++){
        down1[j] = 0.0f;
        float sum = down[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = down[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        down1[j] = sum;
    }
    }
    for (int j = 0; j < N; j++) {
        if(x[j]-down1[j%n1] < -1e-7f || x[j]- up1[j%n2] > 1e-7f){
            g[j] = 0.0f;
        }
        else{
            g[j] = g[j];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }

    free(x);
    free(g);
    free(up1);
    free(down1);
    return;
}

void ecall_reducemax_one(float *input,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*M*L);
    memset(x1, 0, sizeof(float)*M*L);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            float max_val = x[i*C*L+j];
            for (int k = 1; k < C; k++) {
                max_val = fmaxf(max_val, x[i*C*L+j+k*L]);
            }
            x1[i*L+j] = max_val;
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < M*L; j++) {
        float y = (x1[j]/static_cast<float>(M*L)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(M*L)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<M*L*(Ne-1);i++){
        output[i] =(x1[i%(M*L)] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*L*(Ne-1);j<M*L*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*M*L+(j-M*L*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-M*L*(Ne-1)] + sum; 
    }
    free(x);
    free(x1);
	return;
    }

void ecall_reducemax_one_grad(float *input,float *grad,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*M*L);
    memset(x1, 0, sizeof(float)*M*L);
    float *g = (float*)malloc(sizeof(float)*M*L);
    memset(g, 0, sizeof(float)*M*L);
    for(int j=0;j<M*L;j++){
        g[j] = 0.0f;
        float sum = grad[insert*M*L+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*M*L+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            float max_val = x[i*C*L+j];
            for (int k = 1; k < C; k++) {
                max_val = fmaxf(max_val, x[i*C*L+j+k*L]);
            }
            x1[i*L+j] = max_val;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < C; k++) {
                if(fabs(x[i*C*L+j+k*L]-x1[i*L+j]) < 1e-7f){
                    x[i*C*L+j+k*L] = g[i*L+j];
		    for(int kk = k+1; kk < C; kk++){
			x[i*C*L+j+kk*L] = 0.0f;		
		}
		   break;
                }
                else{
                    x[i*C*L+j+k*L] = 0.0f;
                }
            }
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(g);
    free(x1);
    free(x);   
    return;
}

void ecall_huberloss(float* pred,float* real,float* delta, int N,int C,int B, float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = pred[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = pred[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = real[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = real[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }
    int tmp_n = N/B; 
    float delta_tmp;                   
    for (int j = 0; j < N; j++){
        if(C == 1){
           delta_tmp = delta[0];
	}  
        else if(C == B){
	   delta_tmp = delta[j/tmp_n];
	}
        else{
	   delta_tmp = delta[j];
	}
        
        if(fabs(y[j]-x[j]) <= delta_tmp ){
            x[j] = (y[j]-x[j])*(y[j]-x[j])/static_cast<float>(2);            
        }
        else if((y[j]-x[j]) > delta_tmp){
            x[j] = delta_tmp*(y[j]-x[j])-(delta_tmp*delta_tmp/static_cast<float>(2));
        }
        else{
            x[j] = delta_tmp*(x[j]-y[j])-(delta_tmp*delta_tmp/static_cast<float>(2));
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(y);
    return;
}

void ecall_huberloss_grad(float* pred,float* real,float* delta,float* grad, int N,int C,int B, float* output,float* grad_r){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = pred[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = pred[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = real[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = real[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    int tmp_n = N/B;
    float delta_tmp;
    for (int j = 0; j < N; j++){
        if(C == 1){
           delta_tmp = delta[0];
        }
        else if(C == B){
           delta_tmp = delta[j/tmp_n];
        }
        else{
           delta_tmp = delta[j];
        }
        float xx = x[j];
        float yy = y[j];
        if(fabs(yy-xx) <= delta_tmp){
            x[j] = -1*(yy-xx)*g[j];          
            y[j] = (yy-xx)*g[j];  
        }
        else if((yy-xx) > delta_tmp){
            x[j] = -delta_tmp*g[j];
            y[j] = delta_tmp*g[j];  
        }
        else{
            x[j] = delta_tmp*g[j];
            y[j] = -1*delta_tmp*g[j]; 
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float z = (y[j]/static_cast<float>(N)) - c_mean;
        float t = mean + z;
        c_mean = (t - mean) - z;
        mean = t;

        float z2 = (y[j]*y[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + z2;
        c_mean_sqr = (t2 - mean_sqr) - z2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_r[i] =(y[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_r[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = -1.0f * grad_r[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        grad_r[j] =y[j-N*(Ne-1)] + sum; 
    }
    free(x);
    free(y);
    free(g);
    return;
}

void ecall_onehot(float* input, int N, int M,int C,int L,float* output,int alpha){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *x1 = (float*)malloc(sizeof(float)*M*L*alpha);
    memset(x1, 0, sizeof(float)*M*L*alpha);

    int *arg = (int*)malloc(sizeof(int)*M*L);
    memset(arg, 0, sizeof(int)*M*L);
    for(int i = 0;i<M;i++){
        for(int j = 0;j<L;j++){
            arg[i*L+j] = 0;
            float tmpmax = x[i*C*L+0+j];
            for(int k = 0;k<C;k++){
                if(x[i*C*L+k*L+j] > tmpmax){
                    arg[i*L+j] = k;
                    tmpmax = x[i*C*L+k*L+j];
                }
    }
    }
    }

    for(int i = 0;i<M;i++){
        for(int j = 0;j<L;j++){
            for(int k = 0;k<alpha;k++){
                x1[i*L*alpha+j*alpha+k] = 0.0f;
            }}}

    for(int i = 0;i<M;i++){
        for(int j = 0;j<L;j++){
                x1[i*L*alpha+j*alpha+arg[i*L+j]] = 1.0f;
    }}

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < M*L*alpha; j++) {
        float y = (x1[j]/static_cast<float>(M*L*alpha)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(M*L*alpha)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<M*L*alpha*(Ne-1);i++){
        output[i] =(x1[i%M*L*alpha] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*L*alpha*(Ne-1);j<M*L*alpha*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*M*L*alpha+(j-M*L*alpha*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-M*L*alpha*(Ne-1)] + sum; 
    }
    free(x);
    free(x1);
    free(arg);
    return;
    }

void ecall_sample(float* input1,float* input2, int N,int nums, float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = input2[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }
    
    float *z = (float*)malloc(sizeof(float)*N);
    memset(z, 0, sizeof(float)*N);
    for(int num = 0;num<nums;num++){
        for(int i=0;i<N;i++){
            z[i] = gaussrand(x[i], y[i]);
        }
        float mean = 0.0f, std, mean_sqr = 0.0f;
        float c_mean = 0.0f, c_mean_sqr = 0.0f;
        for (int j = 0; j < N; j++) {
            float y = (z[j]/static_cast<float>(N)) - c_mean;
            float t = mean + y;
            c_mean = (t - mean) - y;
            mean = t;

            float y2 = (z[j]*z[j]/static_cast<float>(N)) - c_mean_sqr;
            float t2 = mean_sqr + y2;
            c_mean_sqr = (t2 - mean_sqr) - y2;
            mean_sqr = t2;
        } 
        std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
        mean = mean/static_cast<float>(Nt);
        for(int i=0;i<N*(Ne-1);i++){
            output[num*Ne*N+i] =(z[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
        }
        for(int j=N*(Ne-1);j<N*Ne;j++){
            output[num*Ne*N+j] = 0.0f;
            float sum = 0.0f;
            float c = 0.0f;
            for(int i=0;i<Nt;i++){
                float y = -1.0f * output[num*Ne*N+indexs[i]*N+(j-N*(Ne-1))] - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            output[num*Ne*N+j] =z[j-N*(Ne-1)] + sum; 
        }
    }
    free(x);
    free(y);
    free(z);
    return;
}

void ecall_logprob(float* input,float* mu,float* sigma, int N,float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *m = (float*)malloc(sizeof(float)*N);
    memset(m, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        m[j] = 0.0f;
        float sum = mu[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = mu[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        m[j] = sum;
        }
    float *sig = (float*)malloc(sizeof(float)*N);
    memset(sig, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        sig[j] = 0.0f;
        float sum = sigma[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = sigma[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sig[j] = sum;
        }
    
    for(int i=0;i<N;i++){
        float PI = 3.1415926535897932384626433832795028841971;
        float eps = 1e-11; // small epsilon value
        x[i] = logf(1e-45+(1 / (sqrt(2 * PI) * sig[i]) * exp(-(x[i] - m[i]) * (x[i] - m[i]) / (2 * sig[i]*sig[i]))));
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    free(m);
    free(sig);
    return;
}

void ecall_logprob_grad(float* input, float* grad,float* mu,float* sigma, int N,float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *m = (float*)malloc(sizeof(float)*N);
    memset(m, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        m[j] = 0.0f;
        float sum = mu[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = mu[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        m[j] = sum;
        }
    float *sig = (float*)malloc(sizeof(float)*N);
    memset(sig, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        sig[j] = 0.0f;
        float sum = sigma[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = sigma[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sig[j] = sum;
        }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for(int i=0;i<N;i++){
        g[i] = g[i] * (m[i] - x[i] ) / (sig[i] * sig[i]);
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =g[j-N*(Ne-1)] + sum;
    }
    free(x);
    free(g);
    free(sig);
    free(m);
    return;
}

void ecall_logloss_none(float *input,int N,float *label,float *weight,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = label[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = label[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }

    float epsilon = 1e-7f;
    for (int i = 0; i < N; i++ ){
            x[i] = -1.0f * ((y[i] * logf(x[i] + epsilon) ) + (1 - y[i]) * logf(1 - x[i] + epsilon))* weight[i];
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    free(y);
    return;
}

void ecall_logloss_none_grad(float *input,float *grad,int N,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = label[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = label[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    float epsilon = 1e-11f;
    for (int i = 0; i <N; i++ ){
        float xx = x[i];
        float yy = y[i];
        y[i] = weight[i]*g[i] * (logf(1 - xx+epsilon)-logf(xx+epsilon));
        grad_w[i] = -1 * g[i] * ((yy * logf(xx+epsilon)) + (1 - yy) * logf(1 - xx+epsilon));
        x[i] = weight[i] * g[i] * (xx- yy)/((xx+epsilon)*(1-xx+epsilon));
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_x[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_x[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * grad_x[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        grad_x[j] =x[j-N*(Ne-1)] + sum; 
    }
    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float z = (y[j]/static_cast<float>(N)) - c_mean;
        float t = mean + z;
        c_mean = (t - mean) - z;
        mean = t;

        float z2 = (y[j]*y[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + z2;
        c_mean_sqr = (t2 - mean_sqr) - z2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_l[i] =(y[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_l[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = -1.0f * grad_l[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        grad_l[j] =y[j-N*(Ne-1)] + sum; 
    }

    free(x);
    free(y);
    free(g);
    return;
}

void ecall_reducemean_0(float *input,int N,int M,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int i = 0; i < L; i++){
        output[i] = 0.0f;
        for (int j = 0; j < M; j++){   
            output[i] += x[i+j*L]/static_cast<float>(M);
    }
    }
    free(x);
    return;
}

void ecall_reducemean_0_grad(float *input,float* grad,int N,int M,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    for (int i = 0; i < M; i++){
        for (int j = 0; j < L; j++){   
            x[i*L+j] = grad[j]/static_cast<float>(M);
        }}
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x);
    return;
}
 
void ecall_moments(float* input, int N,int M, int C, int L, float* mean, float* var){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }

    float *mu = (float*)malloc(sizeof(float)*M*C);
    memset(mu, 0, sizeof(float)*M*C);
    float *sig = (float*)malloc(sizeof(float)*M*C);
    memset(sig, 0, sizeof(float)*M*C);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            mu[i*C+j] = 0.0f;
            sig[i*C+j] = 0.0f;
            for (int k = 0; k < L; k++) {
                mu[i*C+j] += x[i*C*L+j*L+k]/static_cast<float>(L);
                sig[i*C+j] += x[i*C*L+j*L+k] * x[i*C*L+j*L+k]/static_cast<float>(L);
    }       
    }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            sig[i*C+j] = sig[i*C+j] - mu[i*C+j]*mu[i*C+j];
        }
    }

    float mean_x = 0.0f, std, mean_x_sqr = 0.0f;
    float c_mean_x = 0.0f, c_mean_x_sqr = 0.0f;
    for (int j = 0; j < M*C; j++) {
        float y = (mu[j]/static_cast<float>(M*C)) - c_mean_x;
        float t = mean_x + y;
        c_mean_x = (t - mean_x) - y;
        mean_x = t;

        float y2 = (mu[j]*mu[j]/static_cast<float>(M*C)) - c_mean_x_sqr;
        float t2 = mean_x_sqr + y2;
        c_mean_x_sqr = (t2 - mean_x_sqr) - y2;
        mean_x_sqr = t2;
    } 
    std = sqrtf(mean_x_sqr - mean_x * mean_x)/static_cast<float>(Nt);
    mean_x = mean_x/static_cast<float>(Nt);
    for(int i=0;i<M*C*(Ne-1);i++){
        mean[i] =(mu[i%M*C] - gaussrand(mean_x, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*C*(Ne-1);j<M*C*Ne;j++){
        mean[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * mean[indexs[i]*M*C+(j-M*C*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        mean[j] =mu[j-M*C*(Ne-1)] + sum; 
    }

    mean_x = 0.0f, std, mean_x_sqr = 0.0f;
    c_mean_x = 0.0f, c_mean_x_sqr = 0.0f;
    for (int j = 0; j < M*C; j++) {
        float y = (sig[j]/static_cast<float>(M*C)) - c_mean_x;
        float t = mean_x + y;
        c_mean_x = (t - mean_x) - y;
        mean_x = t;

        float y2 = (sig[j]*sig[j]/static_cast<float>(M*C)) - c_mean_x_sqr;
        float t2 = mean_x_sqr + y2;
        c_mean_x_sqr = (t2 - mean_x_sqr) - y2;
        mean_x_sqr = t2;
    } 
    std = sqrtf(mean_x_sqr - mean_x * mean_x)/static_cast<float>(Nt);
    mean_x = mean_x/static_cast<float>(Nt);
    for(int i=0;i<M*C*(Ne-1);i++){
        var[i] =(sig[i%M*C] - gaussrand(mean_x, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*C*(Ne-1);j<M*C*Ne;j++){
        var[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * var[indexs[i]*M*C+(j-M*C*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        var[j] =sig[j-M*C*(Ne-1)] + sum; 
    }

    free(x);
    free(mu);
    free(sig);
    return;
}

void ecall_batchnorm(float* input,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* output,float epsilon){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *mu = (float*)malloc(sizeof(float)*M*H);
    memset(mu, 0, sizeof(float)*M*H);
    for(int j=0;j<M*H;j++){
        mu[j] = 0.0f;
        float sum = mean_x[insert*M*H+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = mean_x[indexs[i]*M*H+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        mu[j] = sum;
    }
    float *sig = (float*)malloc(sizeof(float)*M*H);
    memset(sig, 0, sizeof(float)*M*H);
    for(int j=0;j<M*H;j++){
        sig[j] = 0.0f;
        float sum = variance[insert*M*H+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = variance[indexs[i]*M*H+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sig[j] = sum;
    }

    for(int i = 0;i<W;i++){
        for(int j = 0;j<M;j++){
            for(int k = 0;k<H;k++){
                x[j*H*W+k*W+i] = scale[i] * (x[j*H*W+k*W+i] - mu[j*H+k]) / sqrt(sig[j*H+k] + epsilon) + offset[i];
    }}}

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }

    free(x);
    free(mu);
    free(sig);
    return;
    }

void ecall_moments_grad(float* input, float* grad1,float* grad2,int N,int M, int C, int L, float* output){
    float *in = (float*)malloc(sizeof(float)*N);
    memset(in, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        in[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        in[j] = sum;
        
    }
    float *g1 = (float*)malloc(sizeof(float)*M*C);
    memset(g1, 0, sizeof(float)*M*C);
    for(int j=0;j<M*C;j++){
        g1[j] = 0.0f;
        float sum = grad1[insert*M*C+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad1[indexs[i]*M*C+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g1[j] = sum;
    }
    float *g2 = (float*)malloc(sizeof(float)*M*C);
    memset(g2, 0, sizeof(float)*M*C);
    for(int j=0;j<M*C;j++){
        g2[j] = 0.0f;
        float sum = grad2[insert*M*C+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad2[indexs[i]*M*C+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g2[j] = sum;
    }

    float *out = (float*)malloc(sizeof(float)*N);
    memset(out, 0, sizeof(float)*N);

    float *x = (float*)malloc(sizeof(float)*M*C);
    memset(x, 0, sizeof(float)*M*C);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < L; k++) {
                x[i*C+j] += in[i*C*L+j*L+k]/static_cast<float>(L);
                out[i*C*L+j*L+k] = g1[i*C+j]/static_cast<float>(L);
            }}}
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < L; k++) {
                out[i*C*L+j*L+k] += g2[i*C+j] * 2.0f * (in[i*C*L+j*L+k]-x[i*C+j])/static_cast<float>(L);
            }}}

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (out[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (out[j]*out[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(out[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =out[j-N*(Ne-1)] + sum;
    }

    free(out);
    free(g1);
    free(g2);
    free(x);
    return;
}


void ecall_batchnorm_grad(float* input,float* grad,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* grad_in,float* grad_sc,float* grad_off,float* grad_mean,float* grad_var,float epsilon){ 
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        if(fabs(x[j])<1e-7f){
            x[j]= 0.0f;
        }
        }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    float *mu = (float*)malloc(sizeof(float)*M*H);
    memset(mu, 0, sizeof(float)*M*H);
    for(int j=0;j<M*H;j++){
        mu[j] = 0.0f;
        float sum = mean_x[insert*M*H+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = mean_x[indexs[i]*M*H+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        mu[j] = sum;
    }
    float *sig = (float*)malloc(sizeof(float)*M*H);
    memset(sig, 0, sizeof(float)*M*H);
    for(int j=0;j<M*H;j++){
        sig[j] = 0.0f;
        float sum = variance[insert*M*H+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = variance[indexs[i]*M*H+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sig[j] = sum;
    }

    float *g_x = (float*)malloc(sizeof(float)*N);
    memset(g_x, 0, sizeof(float)*N);
    float *g_mu = (float*)malloc(sizeof(float)*M*H);
    memset(g_mu, 0, sizeof(float)*M*H);
    float *g_v = (float*)malloc(sizeof(float)*M*H);
    memset(g_v, 0, sizeof(float)*M*H);

	for(int j = 0;j<M;j++){
            for(int k = 0;k<H;k++){
                for(int i = 0;i<W;i++){
                    g_x[j*H*W+k*W+i] = 0.0f;
                    grad_off[i] = 0.0f;
                    grad_sc[i] = 0.0f;
                    g_v[j*H+k] =0.0f;
                    g_mu[j*H+k] =0.0f;
}}}

	for(int j = 0;j<M;j++){
        for(int k = 0;k<H;k++){
		    for(int i = 0;i<W;i++){
                g_x[j*H*W+k*W+i] += g[j*H*W+k*W+i] * scale[i]/sqrt(sig[j*H+k] + epsilon);
		        grad_off[i] += g[j*H*W+k*W+i];
                grad_sc[i] += g[j*H*W+k*W+i]*((x[j*H*W+k*W+i] - mu[j*H+k]) / sqrt(sig[j*H+k] + epsilon));
		        g_v[j*H+k] += -1.0f*g[j*H*W+k*W+i]*scale[i]*(x[j*H*W+k*W+i]-mu[j*H+k])/(2.0f*sqrt(sig[j*H+k] + epsilon)*(sig[j*H+k] + epsilon));
                g_mu[j*H+k] += -1.0f*g[j*H*W+k*W+i]*scale[i]/sqrt(sig[j*H+k]+epsilon);
    }}}

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (g_x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g_x[j]*g_x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_in[i] =(g_x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_in[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * grad_in[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        grad_in[j] =g_x[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < M*H; j++) {
        float y = (g_mu[j]/static_cast<float>(M*H)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g_mu[j]*g_mu[j]/static_cast<float>(M*H)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<M*H*(Ne-1);i++){
        grad_mean[i] =(g_mu[i%M*H] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*H*(Ne-1);j<M*H*Ne;j++){
        grad_mean[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * grad_mean[indexs[i]*M*H+(j-M*H*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        grad_mean[j] =g_mu[j-M*H*(Ne-1)] + sum; 
    }
    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < M*H; j++) {
        float y = (g_v[j]/static_cast<float>(M*H)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g_v[j]*g_v[j]/static_cast<float>(M*H)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<M*H*(Ne-1);i++){
        grad_var[i] =(g_v[i%M*H] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*H*(Ne-1);j<M*H*Ne;j++){
        grad_var[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * grad_var[indexs[i]*M*H+(j-M*H*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        grad_var[j] =g_v[j-M*H*(Ne-1)] + sum; 
    }

free(g_x);
free(g_v);
free(g_mu);
free(x);
free(g);
free(mu);
free(sig);
return;
}
  
void ecall_dot(float* input1,float* input2, int N1,int N2,int* shape1, int* shape2,int ndims,float* output){
    float *x1 = (float*)malloc(sizeof(float)*N1);
    memset(x1, 0, sizeof(float)*N1);
    for(int j=0;j<N1;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
		
    float *x2 = (float*)malloc(sizeof(float)*N2);
    memset(x2, 0, sizeof(float)*N2);
    for(int j=0;j<N2;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }
    
    if(ndims == 2){
        if(shape1[1] > shape2[1]){
            int M = shape1[0]/Ne;
            int L = shape1[1];
            for(int i=0;i<M;i++){
                for(int j=0;j<L;j++){
                    x1[i*L+j] = x1[i*L+j] * x2[i];
                }
            }
        }
        else if(shape1[1] < shape2[1]){
            int M = shape1[0]/Ne;
            int L = shape2[1];
            for(int i=0;i<M;i++){
                for(int j=0;j<L;j++){
                    x2[i*L+j] = x2[i*L+j] * x1[i];
                }
            }
        }
        else{
            int M = shape1[0]/Ne;
            int L = shape1[1];
            for(int i=0;i<M;i++){
                for(int j=0;j<L;j++){
                    x1[i*L+j] = x1[i*L+j] * x2[i*L+j];
                }
            }
        }
    }
    else{
        if(shape1[1] > shape2[1]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        x1[i*C*L+j*L+k] = x1[i*C*L+j*L+k]*x2[i*L+k];        
                        }
                    }
                }
        }
        else if(shape1[1] < shape2[1]){
            int M = shape1[0]/Ne;
            int C = shape2[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        x2[i*C*L+j*L+k] = x2[i*C*L+j*L+k]*x1[i*L+k];        
                        }
                    }
                }
        }
        else if(shape1[2] > shape2[2]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        x1[i*C*L+j*L+k] = x1[i*C*L+j*L+k]*x2[i*C+j];        
                        }
                    }
                }
        }
        else if(shape1[2] < shape2[2]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape2[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        x2[i*C*L+j*L+k] = x2[i*C*L+j*L+k]*x1[i*C+j];        
                        }
                    }
                }
        }
        else{
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        x1[i*C*L+j*L+k] = x1[i*C*L+j*L+k]*x2[i*C*L+j*L+k];        
                        }
                    }
                }
        }
    }

    if(N1 >= N2){
        float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N1; j++) {
        float y = (x1[j]/static_cast<float>(N1)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N1)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N1*(Ne-1);i++){
        output[i] =(x1[i%N1] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N1*(Ne-1);j<N1*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N1+(j-N1*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N1*(Ne-1)] + sum; 
    }
    }
    else{
        float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N2; j++) {
        float y = (x2[j]/static_cast<float>(N2)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N2)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N2*(Ne-1);i++){
        output[i] =(x2[i%N2] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N2*(Ne-1);j<N2*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N2+(j-N2*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x2[j-N2*(Ne-1)] + sum; 
    }
    }
    free(x1);
    free(x2);
    return;
}

void ecall_dot_grad(float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2 ,int ndims,float* output1,float* output2){
    float *x1 = (float*)malloc(sizeof(float)*N1);
    memset(x1, 0, sizeof(float)*N1);
    for(int j=0;j<N1;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
		
    float *x2 = (float*)malloc(sizeof(float)*N2);
    memset(x2, 0, sizeof(float)*N2);
    for(int j=0;j<N2;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }

    int N = std::max(N1,N2);
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    if(ndims == 2){
        if(shape1[1] > shape2[1]){
            int M = shape1[0]/Ne;
            int L = shape1[1];
            for(int i=0;i<M;i++){
                float tmp2 = x2[i];
                x2[i] = 0.0f;
                for(int j=0;j<L;j++){
                    float tmp1 = x1[i*L+j];
                    x1[i*L+j] = tmp2 * g[i*L+j];
                    x2[i] += tmp1 * g[i*L+j];
                }
            }
        }
        else if(shape1[1] < shape2[1]){
            int M = shape1[0]/Ne;
            int L = shape2[1];
            for(int i=0;i<M;i++){
                float tmp1 = x1[i];
                x1[i] = 0.0f;
                for(int j=0;j<L;j++){
                    float tmp2 = x2[i*L+j];
                    x1[i] += tmp2 * g[i*L+j];
                    x2[i*L+j] = tmp1 * g[i*L+j];
                }
            }
        }
        else{
            int M = shape1[0]/Ne;
            int L = shape1[1];
            for(int i=0;i<M;i++){
                for(int j=0;j<L;j++){
                    float tmp1 = x1[i*L+j];
                    float tmp2 = x2[i*L+j];
                    x1[i*L+j] = tmp2 * g[i*L+j];
                    x2[i*L+j] = tmp1 * g[i*L+j];
                }
            }
        }
    }
    else{
        if(shape1[1] > shape2[1]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int k=0;k<L;k++){
                    float tmp2 = x2[i*L+k];
                    x2[i*L+k] = 0.0f;
                    for(int j=0;j<C;j++){
                        float tmp1 = x1[i*C*L+j*L+k];
                        x2[i*L+k] += tmp1 * g[i*C*L+j*L+k]; 
                        x1[i*C*L+j*L+k] = tmp2 * g[i*C*L+j*L+k];        
                        }
                    }
                }
        }
        else if(shape1[1] < shape2[1]){
            int M = shape1[0]/Ne;
            int C = shape2[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int k=0;k<L;k++){
                    float tmp1 = x1[i*L+k];
                    x1[i*L+k] = 0.0f;
                    for(int j=0;j<C;j++){
                        float tmp2 = x2[i*C*L+j*L+k];
                        x1[i*L+k] += tmp2 * g[i*C*L+j*L+k]; 
                        x2[i*C*L+j*L+k] = tmp1 * g[i*C*L+j*L+k];             
                        }
                    }
                }
        }
        else if(shape1[2] > shape2[2]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    float tmp2 = x2[i*C+j];
                    x2[i*C+j] = 0.0f;
                    for(int k=0;k<L;k++){
                        float tmp1 = x1[i*C*L+j*L+k];
                        x2[i*C+j] += tmp1 * g[i*C*L+j*L+k];
                        x1[i*C*L+j*L+k] = tmp2 * g[i*C*L+j*L+k];           
                        }
                    }
                }
        }
        else if(shape1[2] < shape2[2]){
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape2[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    float tmp1 = x1[i*C+j];
                    x1[i*C+j] = 0.0f;
                    for(int k=0;k<L;k++){
                        float tmp2 = x2[i*C*L+j*L+k];
                        x1[i*C+j] += tmp2 * g[i*C*L+j*L+k];
                        x2[i*C*L+j*L+k] = tmp1 * g[i*C*L+j*L+k];        
                        }
                    }
                }
        }
        else{
            int M = shape1[0]/Ne;
            int C = shape1[1];
            int L = shape1[2];
            for(int i=0;i<M;i++){
                for(int j=0;j<C;j++){
                    for(int k=0;k<L;k++){
                        float tmp1 = x1[i*C*L+j*L+k];
                        float tmp2 = x2[i*C*L+j*L+k];
                        x1[i*C*L+j*L+k] = tmp2 * g[i*C*L+j*L+k]; 
                        x2[i*C*L+j*L+k] = tmp1 * g[i*C*L+j*L+k]; 
                        }
                    }
                }
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N1; j++) {
        float y = (x1[j]/static_cast<float>(N1)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N1)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N1*(Ne-1);i++){
        output1[i] =(x1[i%N1] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N1*(Ne-1);j<N1*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N1+(j-N1*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N1*(Ne-1)] + sum; 
    }
    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N2; j++) {
        float y = (x2[j]/static_cast<float>(N2)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N2)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N2*(Ne-1);i++){
        output2[i] =(x2[i%N2] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N2*(Ne-1);j<N2*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N2+(j-N2*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N2*(Ne-1)] + sum; 
    }
    free(x1);
    free(x2);
    free(g);
    return;
}

void ecall_divi(float* input1,float* input2, int N1,int N2,int* shape1, int* shape2 ,float* output){
    float *x1 = (float*)malloc(sizeof(float)*N1);
    memset(x1, 0, sizeof(float)*N1);
    for(int j=0;j<N1;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
    float *x2 = (float*)malloc(sizeof(float)*N2);
    memset(x2, 0, sizeof(float)*N2);
    for(int j=0;j<N2;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }
    if(shape1[1] > shape2[1]){
        int M = shape1[0]/Ne;
        int C = shape1[1];
        int L = shape1[2];
        for(int i=0;i<M;i++){
            for(int j=0;j<C;j++){
                for(int k=0;k<L;k++){
                    x1[i*C*L+j*L+k] = x1[i*C*L+j*L+k]/(x2[i*L+k]+1e-10f);        
}
        }
        }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N1; j++) {
        float y = (x1[j]/static_cast<float>(N1)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N1)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N1*(Ne-1);i++){
        output[i] =(x1[i%N1] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N1*(Ne-1);j<N1*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N1+(j-N1*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N1*(Ne-1)] + sum; 
    }
    
    }
    else if(shape1[1] < shape2[1]){
        int M = shape2[0]/Ne;
        int C = shape2[1];
        int L = shape2[2];
        for(int i=0;i<M;i++){
            for(int j=0;j<C;j++){
                for(int k=0;k<L;k++){
                    x2[i*C*L+j*L+k] = x1[i*L+k]/(x2[i*C*L+j*L+k]+1e-10f);
        }
        }
        }
        float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N2; j++) {
        float y = (x2[j]/static_cast<float>(N2)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N2)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N2*(Ne-1);i++){
        output[i] =(x2[i%N2] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N2*(Ne-1);j<N2*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N2+(j-N2*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x2[j-N2*(Ne-1)] + sum; 
    }
    }
    else{
        for(int i=0;i<N1;i++){
            x1[i] = x1[i]/(x2[i]+1e-10f);
        }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N1; j++) {
        float y = (x1[j]/static_cast<float>(N1)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N1)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N1*(Ne-1);i++){
        output[i] =(x1[i%N1] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N1*(Ne-1);j<N1*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N1+(j-N1*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N1*(Ne-1)] + sum; 
    }
    }
    free(x1);
    free(x2);
    return;
}

void ecall_divi_grad(float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2 ,float* output1,float* output2){
    float *x1 = (float*)malloc(sizeof(float)*N1);
    memset(x1, 0, sizeof(float)*N1);
    for(int j=0;j<N1;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
    }
		
    float *x2 = (float*)malloc(sizeof(float)*N2);
    memset(x2, 0, sizeof(float)*N2);
    for(int j=0;j<N2;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
    }

    int N = std::max(N1,N2);
    float *g = (float*)malloc(sizeof(float)*N);
        memset(g, 0, sizeof(float)*N);
        for(int j=0;j<N;j++){
            g[j] = 0.0f;
            float sum = grad[insert*N+j];
            float c = 0.0f;
            for(int i=0;i<Nt;i++){
                float y = grad[indexs[i]*N+j] - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            g[j] = sum;
        }

    if(N1 > N2){
        int M = shape1[0]/Ne;
        int C = shape1[1];
        int L = shape1[2];
        for(int i=0;i<M;i++){
            for(int k=0;k<L;k++){
		          float tmp2 = x2[i*L+k];
		          x2[i*L+k] = 0.0f;
                for(int j=0;j<C;j++){
                    float tmp1 = x1[i*C*L+j*L+k];
                    x1[i*C*L+j*L+k] = g[i*C*L+j*L+k]/(tmp2+1e-10f);
                    x2[i*L+k] +=  -1.0f * g[i*C*L+j*L+k] * tmp1/(tmp2*tmp2+1e-10f);       
}
        }
        }
    }
    else if(N1 < N2){
        int M = shape1[0]/Ne;
        int C = shape2[1];
        int L = shape1[2];
        for(int i=0;i<M;i++){
        for(int k=0;k<L;k++){
        float tmp2 = x1[i*L+k];
        x1[i*L+k] = 0.0f;
                for(int j=0;j<C;j++){
                    float tmp1 = x2[i*C*L+j*L+k];
                    x1[i*L+k] +=  g[i*C*L+j*L+k] / (tmp1+1e-10f);
                    x2[i*C*L+j*L+k] = -g[i*C*L+j*L+k]*tmp2/(tmp1*tmp1+1e-10f);
                    
        }
        }
        }
    }
    else{
        for(int i=0;i<N;i++){
            float tmp1 = x1[i];
            float tmp2 = x2[i];
            x1[i] = g[i] / (tmp2+1e-10f);
            x2[i] = -g[i] * tmp1/(tmp2*tmp2+1e-10f);
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N1; j++) {
        float y = (x1[j]/static_cast<float>(N1)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N1)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N1*(Ne-1);i++){
        output1[i] =(x1[i%N1] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N1*(Ne-1);j<N1*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N1+(j-N1*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N1*(Ne-1)] + sum; 
    }
    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N2; j++) {
        float y = (x2[j]/static_cast<float>(N2)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N2)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N2*(Ne-1);i++){
        output2[i] =(x2[i%N2] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N2*(Ne-1);j<N2*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N2+(j-N2*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N2*(Ne-1)] + sum; 
    }
    free(x1);
    free(x2);
    free(g);
    return;
}

void ecall_math(float* input,float num,int N,float* output){
  float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
   for (int j = 0; j < N; j++) {
        x[j]= x[j]+num;
    }
     float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    return; 
}


void ecall_matadd(float* input1,float* input2,int N,int C,float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }
    for (int j = 0; j < N; j++) {
        x[j]= x[j] + input2[j%C];
}
float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum;
    }
    free(x);
    return;
}

void ecall_matadd_grad(float* input1,float* input2,float* grad,int N,int C,float* output1,float* output2){
	float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
	
	for(int i = 0;i<C;i++){
	output2[i] = 0.0f;
}
	for (int j = 0; j < N; j++) {
	output2[j%C] += g[j]; 
}

	
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (g[j] /static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (g[j]*g[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(g[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =g[j-N*(Ne-1)] + sum;
    }
    
    free(g);
    return;
}


void ecall_reducemin_one(float *input,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*M*L);
    memset(x1, 0, sizeof(float)*M*L);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            float min_val = x[i*C*L+j];
            for (int k = 1; k < C; k++) {
                min_val = fminf(min_val, x[i*C*L+j+k*L]);
            }
            x1[i*L+j] = min_val;
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < M*L; j++) {
        float y = (x1[j]/static_cast<float>(M*L)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(M*L)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<M*L*(Ne-1);i++){
        output[i] =(x1[i%(M*L)] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=M*L*(Ne-1);j<M*L*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*M*L+(j-M*L*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-M*L*(Ne-1)] + sum; 
    }
    free(x);
    free(x1);
	return;
    }


void ecall_reducemin_one_grad(float *input,float *grad,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*M*L);
    memset(x1, 0, sizeof(float)*M*L);
    float *g = (float*)malloc(sizeof(float)*M*L);
    memset(g, 0, sizeof(float)*M*L);
    for(int j=0;j<M*L;j++){
        g[j] = 0.0f;
        float sum = grad[insert*M*L+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*M*L+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            float min_val = x[i*C*L+j];
            for (int k = 1; k < C; k++) {
                min_val = fminf(min_val, x[i*C*L+j+k*L]);
            }
            x1[i*L+j] = min_val;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < C; k++) {
                if(fabs(x[i*C*L+j+k*L]-x1[i*L+j]) < 1e-7f){
                    x[i*C*L+j+k*L] = g[i*L+j];
		    for(int kk = k+1; kk < C; kk++){
			x[i*C*L+j+kk*L] = 0.0f;		
		}
		   break;
                }
                else{
                    x[i*C*L+j+k*L] = 0.0f;
                }
            }
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(g);
    free(x1);
    free(x);   
    return;
}

void ecall_reducemin_zero(float *input,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            float min_val = x[i*L+j];
            for (int k = 1; k < M; k++) {
                min_val = fminf(min_val, x[i*L+j+k*C*L]);
            }
            output[i*L+j] = min_val;
        }
    }
    free(x);
	return;
    }

void ecall_reducemin_zero_grad(float *input,float *grad,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*C*L);
    memset(x1, 0, sizeof(float)*C*L);
    
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            float min_val = x[i*L+j];
            for (int k = 1; k < M; k++) {
                min_val = fminf(min_val, x[i*L+j+k*C*L]);
            }
            x1[i*L+j] = min_val;
        }
    }

    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < M; k++) {
                if(fabs(x[i*L+j+k*C*L]-x1[i*L+j]) < 1e-7f){
                    x[i*L+j+k*C*L] = grad[i*L+j];
		        for(int kk = k+1; kk < M; kk++){
			        x[i*L+j+kk*C*L] = 0.0f;		
		        }
		        break;
                }
                else{
                    x[i*L+j+k*C*L] = 0.0f;
                }
            }
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x1);
    free(x);   
    return;
}

void ecall_reducemax_zero(float *input,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }

    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            float max_val = x[i*L+j];
            for (int k = 1; k < M; k++) {
                max_val = fmaxf(max_val, x[i*L+j+k*C*L]);
            }
            output[i*L+j] = max_val;
        }
    }
    free(x);
	return;
    }

void ecall_reducemax_zero_grad(float *input,float *grad,int N,int M,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        
    }
    float *x1 = (float*)malloc(sizeof(float)*C*L);
    memset(x1, 0, sizeof(float)*C*L);
    
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            float max_val = x[i*L+j];
            for (int k = 1; k < M; k++) {
                max_val = fmaxf(max_val, x[i*L+j+k*C*L]);
            }
            x1[i*L+j] = max_val;
        }
    }

    for (int i = 0; i < C; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < M; k++) {
                if(fabs(x[i*L+j+k*C*L]-x1[i*L+j]) < 1e-7f){
                    x[i*L+j+k*C*L] = grad[i*L+j];
		        for(int kk = k+1; kk < M; kk++){
			        x[i*L+j+kk*C*L] = 0.0f;		
		        }
		        break;
                }
                else{
                    x[i*L+j+k*C*L] = 0.0f;
                }
            }
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }
    free(x1);
    free(x);   
    return;
}

bool compare(int a, int b) {
    return a > b;
};

bool compare_grad(const Element& a, const Element& b) {
    return a.value > b.value;
};


void ecall_sort_de(float *input,int N,int M ,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;    
    }

    float *tmp = (float*)malloc(sizeof(float)*C);
    memset(tmp, 0, sizeof(float)*C);

    for(int i=0;i<M;i++){
        for(int k =0;k<L;k++){
            for(int j =0;j<C;j++){
                tmp[j] = x[i*C*L+j*L+k];
            }
            std::sort(tmp,tmp+C,compare);
            for(int j =0;j<C;j++){
                x[i*C*L+j*L+k] = tmp[j];
            }
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }

    free(x);
    free(tmp);
	return;
}

void ecall_sort_de_grad(float *input,float* grad,int N,int M ,int C,int L,float *output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = input[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;    
    }

    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    Element ele[C];
    for(int i=0;i<M;i++){
        for(int k =0;k<L;k++){
            for(int j =0;j<C;j++){
                ele[j].value = x[i*C*L+j*L+k];
                ele[j].index = j;
            }
            std::sort(ele,ele+C,compare_grad);
            for(int j =0;j<C;j++){
              //x[i*C*L+j*L+k] = 1.0f;
		x[i*C*L+j*L+k] = g[i*C*L+ele[j].index*L+k];
            }
        }
    }	

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;

    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }

    free(x);
    free(g);
	return;
}

void ecall_where_equal(float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
    float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }

    for(int i=0;i<N;i++){
	//x1[i] = x2[i];
        if(fabs(x3[i%n1]-x4[i%n2]) > 1e-7f){
            x1[i] = x2[i];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x2);
    free(x3);
    free(x4);
    return;
}

void ecall_where_equal_grad(float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
    float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for(int i=0;i<N;i++){
        if(fabs(x3[i%n1]-x4[i%n2]) > 1e-7f){
            x1[i] = 0.0f;
            x2[i] = g[i];
        }
        else{
            x1[i] = g[i];
            x2[i] = 0.0f;
        }
    }
    
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f;    
    std = 0.0f; 
    mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x2[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output2[i] =(x2[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N*(Ne-1)] + sum; 
    }



    free(x1);
    free(x2);
    free(x3);
    free(x4);
    free(g);
    return;
}


void ecall_where_gequal(float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
    float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }

    for(int i=0;i<N;i++){
        if(x3[i%n1] - x4[i%n2] < 1e-7f){
            x1[i] = x2[i];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x3);
    free(x2);
    free(x4);
    return;
}

void ecall_where_gequal_grad(float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
    float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for(int i=0;i<N;i++){
        if(x3[i%n1]-x4[i%n2] <0.0f){
            x1[i] = 0.0f;
            x2[i] = g[i];
        }
        else{
            x1[i] = g[i];
            x2[i] = 0.0f;
        }
    }
    
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f;    
    std = 0.0f; 
    mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x2[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output2[i] =(x2[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x2);
    free(x3);
    free(x4);
    free(g);
    return;
}

void ecall_where_greater(float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
   float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }

    for(int i=0;i<N;i++){
        if(x3[i%n1] - x4[i%n2] <= 1e-7f){
            x1[i] = x2[i];
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x1[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x3);
    free(x2);
    free(x4);
    return;
}

void ecall_where_greater_grad(float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
    float *x1 = (float*)malloc(sizeof(float)*N);
    memset(x1, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x1[j] = 0.0f;
        float sum = input1[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input1[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x1[j] = sum;
        }
    float *x2 = (float*)malloc(sizeof(float)*N);
    memset(x2, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x2[j] = 0.0f;
        float sum = input2[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input2[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x2[j] = sum;
        }
    float *x3 = (float*)malloc(sizeof(float)*n1);
    memset(x3, 0, sizeof(float)*n1);
    if(n1 == 1){
        x3[0] = cond1[0];
    }
    else{
        for(int j=0;j<n1;j++){
        x3[j] = 0.0f;
        float sum = cond1[insert*n1+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond1[indexs[i]*n1+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x3[j] = sum;
        }
    }
    
    float *x4 = (float*)malloc(sizeof(float)*n2);
    memset(x4, 0, sizeof(float)*n2);
    if(n2 == 1){
        x4[0] = cond2[0];
    }
    else{
        for(int j=0;j<n2;j++){
        x4[j] = 0.0f;
        float sum = cond2[insert*n2+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = cond2[indexs[i]*n2+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x4[j] = sum;
        }
    }
    float *g = (float*)malloc(sizeof(float)*N);
    memset(g, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }

    for(int i=0;i<N;i++){
        if(x3[i%n1]-x4[i%n2] <= 1e-7f){
            x1[i] = 0.0f;
            x2[i] = g[i];
        }
        else{
            x1[i] = g[i];
            x2[i] = 0.0f;
        }
    }
    
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x1[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x1[j]*x1[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output1[i] =(x1[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x1[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f;    
    std = 0.0f; 
    mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x2[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x2[j]*x2[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output2[i] =(x2[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output2[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output2[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output2[j] =x2[j-N*(Ne-1)] + sum; 
    }

    free(x1);
    free(x2);
    free(x3);
    free(x4);
    free(g);
    return;
}

void ecall_softmax_cross_entropy(float* pred,int N,int M,int C,float* real,float* output){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = pred[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = pred[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = real[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = real[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }

    for (int i = 0; i < M; i++) {
            float max_val = x[i*C];
            for (int j = 1; j < C; j++) {
                max_val = fmaxf(max_val, x[i*C+j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < C; j++) {
                sum += expf(x[i*C+j] - max_val);
            }
            for (int j = 0; j < C; j++) {
                x[i*C+j] = expf(x[i*C+j] - max_val) / sum;
            }
        }
    
    const float NEAR_0 = 1e-11;
    for (int i = 0; i < M; i++) {
      output[i] = 0;
      for (int j = 0; j <C; j++) {
            output[i] -= y[i*C+j] * logf(x[i*C+j]+NEAR_0);
      }
    }

    free(x);
    free(y);
    return;
}

void ecall_softmax_cross_entropy_grad(float* pred,float* grad, int N, int M,int C,float* real,float* output,float* grad_r){
    float *x = (float*)malloc(sizeof(float)*N);
    memset(x, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        x[j] = 0.0f;
        float sum = pred[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = pred[indexs[i]*N+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
        }

    float *y = (float*)malloc(sizeof(float)*N);
    memset(y, 0, sizeof(float)*N);
    for(int j=0;j<N;j++){
        y[j] = 0.0f;
        float sum = real[insert*N+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = real[indexs[i]*N+j] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        y[j] = sum;
    }

    for (int i = 0; i < M; i++) {
            float max_val = x[i*C];
            for (int j = 1; j < C; j++) {
                max_val = fmaxf(max_val, x[i*C+j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < C; j++) {
                sum += expf(x[i*C+j] - max_val);
            }
            for (int j = 0; j < C; j++) {
                x[i*C+j] = expf(x[i*C+j] - max_val) / sum;
            }
        }
    
    for(int i=0;i<M;i++){
        for(int j = 0;j<C;j++){
           float xx = x[i*C+j];
           float yy = y[i*C+j];
           x[i*C+j] = grad[i] * (xx-yy);
           y[i*C+j] = -grad[i] * (logf(xx));
        }
    }

    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float y = (x[j]/static_cast<float>(N)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        output[i] =(x[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =x[j-N*(Ne-1)] + sum; 
    }

    mean = 0.0f, std, mean_sqr = 0.0f;
    c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N; j++) {
        float z = (y[j]/static_cast<float>(N)) - c_mean;
        float t = mean + z;
        c_mean = (t - mean) - z;
        mean = t;

        float z2 = (y[j]*y[j]/static_cast<float>(N)) - c_mean_sqr;
        float t2 = mean_sqr + z2;
        c_mean_sqr = (t2 - mean_sqr) - z2;
        mean_sqr = t2;
    } 
    std = sqrtf(mean_sqr - mean * mean)/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*(Ne-1);i++){
        grad_r[i] =(y[i%N] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*(Ne-1);j<N*Ne;j++){
        grad_r[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float z = -1.0f * grad_r[indexs[i]*N+(j-N*(Ne-1))] - c;
            float t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        grad_r[j] =y[j-N*(Ne-1)] + sum; 
    }
    free(x);
    free(y);
    return;
}


void ecall_matmul(float *input,float* weight,int N,int M,int C,float *output){
    float *x = (float*)malloc(sizeof(float)*N*M);
    memset(x, 0, sizeof(float)*N*M);
    for(int j=0;j<N*M;j++){
        x[j] = 0.0f;
        float sum = input[insert*N*M+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N*M+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }
    float *z = (float*)malloc(sizeof(float)*N*C);
    memset(z, 0, sizeof(float)*N*C);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < C; j++){
            z[i*C+j] = 0.0f;
            for(int k=0;k<M; k++){
                z[i*C+j] += x[i*M+k]*weight[k*C+j];
            }
        }
    }
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N*C; j++) {
        float y = (z[j]/static_cast<float>(N*C)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (z[j]*z[j]/static_cast<float>(N*C)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*C*(Ne-1);i++){
        output[i] =(z[i%(N*C)] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*C*(Ne-1);j<N*C*Ne;j++){
        output[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output[indexs[i]*N*C+(j-N*C*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output[j] =z[j-N*C*(Ne-1)] + sum; 
    }
    free(x);
    free(z);
    return;
}

void ecall_matmul_grad(float *input,float* grad,float* weight,int N,int M,int C,float *output1,float* output2){
    float *x = (float*)malloc(sizeof(float)*N*M);
    memset(x, 0, sizeof(float)*N*M);
    for(int j=0;j<N*M;j++){
        x[j] = 0.0f;
        float sum = input[insert*N*M+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = input[indexs[i]*N*M+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        x[j] = sum;
    }

    float *g = (float*)malloc(sizeof(float)*N*C);
    memset(g, 0, sizeof(float)*N*C);
    for(int j=0;j<N*C;j++){
        g[j] = 0.0f;
        float sum = grad[insert*N*C+j];
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = grad[indexs[i]*N*C+j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        g[j] = sum;
    }
    
    for (int i = 0; i < M; i++){
        for (int j = 0; j < C; j++){
            output2[i*C+j] = 0.0f;
            for(int k=0;k<N; k++){
                output2[i*C+j] += g[k*C+j]*x[k*M+i];
            }
        }
    }

    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            x[i*M+j] = 0.0f;
            for(int k=0;k<C; k++){
                x[i*M+j] += g[i*C+k]*weight[j*C+k];
            }
        }
    }
    
    float mean = 0.0f, std, mean_sqr = 0.0f;
    float c_mean = 0.0f, c_mean_sqr = 0.0f;
    for (int j = 0; j < N*M; j++) {
        float y = (x[j]/static_cast<float>(N*M)) - c_mean;
        float t = mean + y;
        c_mean = (t - mean) - y;
        mean = t;

        float y2 = (x[j]*x[j]/static_cast<float>(M*N)) - c_mean_sqr;
        float t2 = mean_sqr + y2;
        c_mean_sqr = (t2 - mean_sqr) - y2;
        mean_sqr = t2;
    } 
    std = sqrtf(fabs(mean_sqr - mean * mean))/static_cast<float>(Nt);
    mean = mean/static_cast<float>(Nt);
    for(int i=0;i<N*M*(Ne-1);i++){
        output1[i] =(x[i%(N*M)] - gaussrand(mean, std))/static_cast<float>(Nt+1);    
    }
    for(int j=N*M*(Ne-1);j<N*M*Ne;j++){
        output1[j] = 0.0f;
        float sum = 0.0f;
        float c = 0.0f;
        for(int i=0;i<Nt;i++){
            float y = -1.0f * output1[indexs[i]*N*M+(j-N*M*(Ne-1))] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        output1[j] =x[j-N*M*(Ne-1)] + sum; 
    }

    free(x);
    free(g);
    return;
}
