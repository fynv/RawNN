const static size_t layer20_mean_size = 1024;
static unsigned layer20_mean[]={
0x3f266ee1,0xbeb43a24,0xbe1d1c37,0xbe7a567d,0xbd97a1de,0xbf40a362,0xbe86f30d,0xbef21521,0xbdd8f59a,0x3e1b7936,
0xbf5573e1,0xbea975eb,0x3ef6fccd,0xbf43d046,0xbee9b37c,0x3dfd398a,0xbe16cc2d,0xbd3293f1,0xbf011cee,0xbedcf1bf,
0xbe9cbefc,0x3e3938c2,0xbea98664,0xbdb55ea9,0x3d7fc0fa,0xbf0bb6b7,0x3e23ee7d,0xbeb8392a,0xbef81bb6,0xbe91027e,
0xbd164104,0xbd9745f9,0xbeed074e,0xbecb3083,0x3d659bdd,0xbeef5bf0,0xbe391602,0x3dd81241,0xbe4f306e,0xbc8e5187,
0xbe54dfaf,0xbeb4c932,0xbed29579,0xbed83c04,0xbf2041bb,0xbe824799,0xbd14c6e6,0x3cf408b1,0xbe1ac145,0x3eb3245d,
0x3e8301a9,0x3d4fa499,0x3da96700,0xbe36f9ad,0x3cf83129,0xbe526c2f,0xbe64152d,0xbdf04cfc,0xbe9adbff,0xbc8addcc,
0xbea47542,0xbe94f2de,0x3e360904,0x3d725194,0xbf06e989,0xbed18b91,0xbd8e685f,0xbda37d0f,0xbece1dfd,0x3f15d9ae,
0xbe501058,0x3de090dd,0x3e9a5c72,0xbebf4ba3,0xbe0fb27e,0xbe451596,0x3d30364a,0x3de72091,0xbe963392,0xbeb57810,
0xbea8a104,0xbe135438,0x3d88ea51,0x3e8a9939,0xbefa2fc3,0xbddad0ab,0x3e40a395,0xbf696221,0xbec14e29,0xbd1ef029,
0xbe7ded1c,0x3e052715,0x3dd699f8,0xbe3c2769,0x3be05b09,0x3d3bc5fa,0xbeb10f50,0xbe46b862,0xbe821046,0xbf0406fb,
0x3b067c90,0xbe888ff5,0xbe9a17fd,0xbe1daf0c,0xbe2287d0,0xbe618c29,0xbed743cf,0xbeab2252,0x3c23bbc1,0xbdb8ead4,
0xbe9d5598,0xbe428067,0x3f10e79f,0xbed8ed38,0xbe6465bd,0x3ea60126,0xbd4fe5b4,0x3ebbd7ad,0xbf272bc7,0xbecabc0f,
0xbeb9e152,0xbe742c6e,0xbeefc36d,0xbecaebd7,0x3dc7ff05,0xbec8a21b,0xbf884d8f,0xbe9b0ebc,0xbe871fbc,0xbe0c6008,
0xbdd707e3,0xbe3babcd,0xbf41e124,0x3de3bc6b,0xbd93ec25,0x3ce727de,0x3bd9e63b,0x3bef9189,0xbec6590e,0xbe8feac8,
0x3e16347b,0xbbbc806a,0x3d8f3b5a,0xbee666b3,0x3e6cdd0e,0xbe1ce47a,0xbcc81495,0xbf536469,0xbe57eebe,0x3f3b2c3d,
0xbeb87f72,0xbebd1fe2,0x3ecde9a8,0xbed9673f,0xbe1abf32,0xbe177ace,0xbec3e122,0x3dd00055,0xbf0fc6b2,0xbe1b7fef,
0x3dac9770,0xbe4e2ed0,0x3d91d5c3,0xbf09bb00,0xbef05ded,0x3ede1a3b,0xbeacfbbb,0xbee379be,0xbd951ce5,0x3e9d5116,
0x3e473d46,0xbf1b10b5,0xbe707010,0xbee37ff8,0xbd0b47e8,0xbe1d7208,0xbe888f0d,0x3ec7f254,0x3e2ec0ec,0xbf040e7c,
0xbf045cab,0x3da4639b,0x3e57a382,0xbf1bd801,0xbf141c53,0xbea84d77,0x3f62a20a,0xbc3ff116,0xbdb97b5b,0xbf1f53ee,
0x3f5eb9f5,0xbf208633,0x3ddbc582,0x3dd8e39c,0xbefdec8d,0xbe2e2b72,0x3ec1209a,0x3e249c5f,0x3e045be5,0xbe5d12e5,
0x3eb3679f,0x3e130684,0xbe945e2e,0xbdb9d04b,0x3e21433f,0xbf39cca1,0xbed6ec43,0x3d4eb400,0xbb4d0c7c,0xbd9b494b,
0xbebe35fb,0xbefe5745,0xbebd0891,0xbf0ce718,0x3e9faf0e,0x3d5d6e2a,0xbe08e225,0xbf1e6f2a,0xbedfe9dd,0x3ef705e6,
0x3d5c351e,0x3de4cf5b,0xbee2b444,0xbe9b9834,0xbef716d1,0xbe8be972,0x3e12bf7b,0xbe8a6a53,0xbe8957b4,0x3e3d594b,
0xbdce86c1,0x3e7f9214,0xbdfc5f2c,0xbdf78cff,0xbeb54fcc,0x3e8ff56b,0xbe818108,0xbe5b190d,0xbdc906ca,0xbea3c163,
0xbe78f86c,0xbe84e5a3,0x3ed7ed79,0x3e7eb271,0xbee362f7,0xbeb29fa3,0xbdd2fa16,0xbec68fe5,0x3d6fc7c9,0xba92daac,
0x3e766f99,0xbf02f9d1,0xbece8c79,0xbecc1a9b,0xbee37afb,0xbe8ee417,};

