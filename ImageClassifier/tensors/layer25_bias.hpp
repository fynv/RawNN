const static size_t layer25_bias_size = 1024;
static unsigned layer25_bias[]={
0xbe05c957,0xbe353b85,0xbe21b16a,0xbeb28875,0xbe007781,0xbd9bb5fc,0xbeb1cf39,0xbeeacf32,0xbdc4f261,0xbd0061dc,
0x3e236bc8,0xbe16f8c8,0xbe6dae53,0xbe26a8c5,0xbec48ab8,0x3ad07d87,0xbe1dfa24,0xbe3272f7,0xbb83cb63,0xbe5458a3,
0xbe65a095,0xbe745970,0xbe5e366e,0xbe9986fe,0xbdef62dc,0xbd2ed322,0xbe0d3dd8,0xbd87d6a6,0xbe7720e6,0xbeaf95cf,
0xbe1040f6,0xbe139847,0xbe4e9c9b,0x3dfb7eb8,0xbdf0df0c,0xbe7063ac,0x3d192bb8,0xbdd1e476,0xbe53f1ee,0xbdbb0433,
0xbe65d0f4,0xbe6f1b46,0xbea761ef,0xbe9116a7,0xbe46ff6b,0xbdc3e588,0xbe8cb483,0xbdb6ba1a,0x3c72b9e6,0xbe8c63a3,
0xbd1ec0f1,0xbeb9ed95,0x3c3d1ac1,0x3dcd0fcc,0xbe4a1cf9,0xbe42d392,0xbd6cac1c,0xbe193c6b,0xbe5c227d,0xbea76944,
0xbdd784e8,0xbda9226c,0xbe6ed7f2,0xbe599f04,0x3d875765,0xbe827e11,0xbdbb62aa,0xbc993f3b,0xbe3815b1,0xbdcbce79,
0xbea68e7f,0xbe4163ba,0xbe816706,0xbe94e0ed,0xbdb4e0fb,0xbe2161ea,0x3ed5ae70,0xbe52ac5c,0xbe8df07b,0xbe381d46,
0xbe56001c,0xbe9a898e,0xbe1fa507,0xbe2ab321,0x3ccb555e,0x3ccafd59,0xbe9730f7,0xbc8c206a,0xbb094327,0xbe2d6cb4,
0xbe84043d,0xbe2079f7,0xbe9023c5,0xbe194006,0xbe5115b6,0x3a7dffbc,0xbeafb6a2,0xbe23dbb0,0xbe09a0cc,0xbe28ed93,
0xbe2be6c8,0xbea0246a,0xbe8c47f9,0xbe8de5b7,0xbe1993d3,0xbe23fe27,0x3d39c464,0xbe0e2c4c,0xbd66a68c,0x3ca24d8b,
0xbe0304f9,0xbde7f497,0xbe4843ec,0xbea03f92,0xbe85a1fb,0xbdc9e35f,0xbe486037,0xbb6f67f2,0xbe530a98,0x3dc4ae2a,
0xbe950e88,0xbc98478b,0xbe324bd2,0xbee1e4bb,0x3e64b2a4,0xbe9980eb,0xbe3654fd,0xbe3afecb,0xbe4035fe,0xbdb3d9ce,
0xbe8e5023,0xbe35f570,0xbea2b754,0xbdf7d2aa,0xbec62c44,0xbe181909,0xbd9784db,0xbe865b0d,0xbe828d90,0xbdc015ab,
0xbd8d55bc,0xbe205bd2,0xbe712fc5,0xbdea1778,0xbe45b247,0xbdb08475,0xbe6dc7cb,0xbed4b957,0xbe78d47f,0xbe9ec680,
0x3cca51b9,0xbdb06c5a,0x3e1fe932,0xbe4362a1,0xbe8aea41,0xbdf9ee59,0xbe852c67,0xbe829cb2,0xbd9b0378,0x3d60abf9,
0xbedcd953,0xbe5b746a,0xbbcb216b,0xbd3f73ee,0xbdf1e13f,0x3d188261,0xbe0cae4b,0xbebd5ec9,0xbeae1df5,0xbe55ee65,
0xbe3164b2,0xbe6c2d71,0xbe7b46a9,0xbdef5502,0xbe4b6edd,0xbe2a3bc0,0xbe510317,0xbe546969,0xbe623264,0xbdc5a328,
0xbdfc4abd,0xbe3afb15,0xbd84a6f1,0xbe19d75a,0xbefb0046,0xbea89310,0xbdd12a50,0xbecbdeb0,0xbd55703d,0x3d79b2f9,
0xbe06b098,0xbe2c2f4e,0xbe5c550e,0xbe2417a9,0xbe42de02,0xbd939ee9,0xbcd67380,0xbe9de01f,0xbe72dc6b,0xbe77806a,
0xbdd0cd47,0xbe347b90,0xbe0d3a43,0xbeb7884c,0xbe145180,0xbe09f5f4,0xbe6df6d4,0xbe181807,0xbdfff2d2,0xbd3f98aa,
0xbe66131e,0xbe9ebf72,0xbe1edbde,0xbe033189,0xbe4909e1,0xbdd8c99e,0xbdef387c,0xbd3b6db1,0xbc80b388,0xbe93bdfb,
0xbe21e3e7,0x3d89b2cf,0xbe060414,0xbe4af654,0xbe999f78,0xbae008fd,0xbd90c2ae,0xbd4480dd,0xbd2f89c1,0xbd27a6d0,
0xbe48537a,0xbe7b0016,0xbe59df69,0xbe418a46,0xbe14ccf2,0xbe49e9b8,0xbd889daf,0xbd4e8e80,0xbebdfb86,0xbea9770c,
0xbe881b0f,0x3d24990c,0xbecb05e1,0xbd27becb,0xbe3597a1,0xbe20da4b,0xbe265f78,0xbde0f058,0xbe5d3f43,0xbdc0a141,
0xbe43a118,0xbc0b1328,0xbc952b7f,0x3d5615dc,0xbe8457c4,0x3d3b4223,};

