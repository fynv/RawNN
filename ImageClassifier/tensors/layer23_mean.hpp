const static size_t layer23_mean_size = 1024;
static unsigned layer23_mean[]={
0xbf81107e,0xbf5131f4,0x3ea5e63d,0xc0362be5,0xbfd1cb3c,0x3ee1976a,0xc00a67b6,0xbfa3917e,0xbd554115,0xbea803bc,
0xbfad374e,0x3f0d2ea1,0xbf9f31ad,0xbf9585d2,0xc02125cf,0xbe8c7f1b,0xc00171c1,0x3cb5a03f,0xbe4ab1cd,0xbf885c66,
0xc002d022,0xbfca5e0c,0x3ea90d80,0xbf8598ac,0xbfdbc7d1,0xbfd82591,0xbda2aebb,0x3fc0ac77,0xbee37719,0xbf911326,
0x3f9890e1,0xbf6d7af8,0xbfb03fd5,0xc00196d2,0xbfc30a53,0xbf5aac1f,0xbfb3c28d,0xbf3e42b6,0xbed9b19c,0xbfcd46c2,
0xbf90dd3f,0xbfa089ab,0xbff98d17,0xbfbb039a,0xc00938ba,0xbf8cc63f,0xbfdf429f,0xc019a490,0xbe8768f0,0xbf75e221,
0xbe60e64d,0xbf915f6e,0xc02ebe11,0xbe414321,0xbeced74e,0xbeef1dc4,0xbf917c3f,0x3f306a2f,0xbf2a86e3,0xbfdc8223,
0xbf657c40,0xbfa7592a,0xbea331c0,0xbfbe5ae0,0xbeacca3d,0xc0084615,0xc0108223,0xbfc52dff,0xbfacf3dc,0xbf685ee8,
0xbfb3b0e0,0xbf017a7f,0xbf77faf8,0xbf69ac97,0xbfcc3005,0xbfbad5f5,0xbf0d5de4,0xc025cdde,0xbf0e7c0c,0xbf3b4840,
0xbffb2139,0xbfd6dbb6,0x3f8761b6,0xbfb993de,0xbf270377,0x3eda58f8,0xbedadab5,0xbfc9d7fd,0xbed7be37,0xbfad7ae3,
0xbfca1b43,0xbf6cb700,0xbfb52fbd,0xc014dbb1,0x3ee125f1,0xbff45282,0xbf989f9b,0x3c1d47e8,0xbea77769,0xc0265645,
0xbf0a7f05,0x3e0483ce,0xc01b0974,0xbf93dd9f,0x3e5807b4,0xbf8d39b8,0xbfdf7501,0xbfcc7748,0xbf8248c1,0xbdc37b51,
0xbfb4ad99,0x3e2af6b0,0xbfaf78d8,0xbe39f2f5,0xbfed0aff,0xbfadcd4e,0xbf543496,0xbf79f9ba,0xc023fe16,0xbe0dcf15,
0xbfd002ed,0xbfec7729,0xbf0d47b8,0xbb9c6c31,0xbfdb2c5b,0xbea00b4b,0xbfabad9f,0xbf86b59f,0xbf8666eb,0xbf6b9c05,
0xbfa05961,0xbfaa2076,0xbf836a46,0xbf65069f,0xbd020be2,0xc01b6151,0xbf72671b,0xbfb35142,0xbfa80493,0xbf2a7497,
0xbdab7822,0x3eddcfbd,0xbf1b974a,0xc0067007,0xbfb3f727,0xbf1cedc7,0xbff43d44,0xc0025df8,0xbfdbd5c5,0xbe8b53da,
0xbe7f0180,0xbe787834,0xbfeb9e9c,0xbfb5a6ae,0xbfbd9233,0xbfb013cf,0xbfd53680,0xbfe0c42b,0xc01af235,0xbeffe20e,
0xbf705a24,0xbf92336f,0xbfa1fb24,0xbf449d6d,0xbf93ab82,0x3f39cbfd,0xbf95264a,0xbf892b78,0xbf7d4f20,0xbf82a1a2,
0xbff81c62,0xbf8f0834,0xc01bc5e1,0xbfda610a,0xbfd2b574,0xbf921e7f,0xc016e79b,0xbfc54ffe,0xbd99cdb1,0xbf6657e2,
0xbfdeed3b,0xbebb1266,0xbfac7090,0xbfbe592d,0x3eec4743,0xbf6d15e4,0xbf6afc38,0xc003cd3a,0xbfe03925,0xbeddeeee,
0xbfeb9770,0xc0339e76,0xbfcda7b1,0xbf9fe03d,0xc0117fa1,0xbf328e4b,0xbf45a19d,0xbf67f39a,0xbfc20aad,0xc00fc305,
0xbdb4f454,0xc023634b,0xbf53574c,0xc008adf0,0x3eeb5069,0xbf081e93,0xbf0c65e3,0xbfd5ee36,0xbfcdc1c6,0xbf0a59d9,
0xbf60fbcb,0xbf20f785,0xbfe3404d,0xbffb5a0f,0xbfc2c263,0x3cdd0d16,0xbd69f7c8,0xbf76fe64,0xbedc400b,0xc0052b78,
0xbf0fa6c0,0xc01597d2,0xbe7e06a8,0x3fa6f361,0xbfe80dc0,0xbfa0b8d7,0xc02cdb77,0x3df594fc,0xbf8b564b,0xbfd8fade,
0xbfbf73d2,0xbf95a473,0xc001e7c0,0x3fa81b6b,0xbfde4295,0xbe8bbebf,0xbec3b5ca,0xbfb128f3,0xbf68b794,0xbf8fa76c,
0xbff06dde,0xbfb052c1,0xbfd06601,0xc022537f,0xbfc2c7f1,0xc00f6439,0xbf0116a2,0x3f1b523c,0x3ea081af,0xbf29c074,
0xbf5a55b6,0xbef2146e,0xbf702a33,0xbf449bdf,0xbf185e59,0xbf54d599,};
