2024-03-15 08:40:14,126 INFO: Start time: 2024-03-15_08-40-14
2024-03-15 08:40:15,905 INFO: Device: cuda

2024-03-15 08:40:15.906213 - Querying celegans SPARQL endpoint with the following queries : ['molecular-entity', 'phenotype', 'interaction', 'disease_plus_ortho', 'disease-ontology', 'phenotype-ontology'].
Querying SPARQL endpoint...:   0%|          | 0/6 [00:00<?, ?it/s]Querying SPARQL endpoint...:  17%|█▋        | 1/6 [00:06<00:32,  6.48s/it]Querying SPARQL endpoint...:  33%|███▎      | 2/6 [00:35<01:18, 19.74s/it]Querying SPARQL endpoint...:  50%|█████     | 3/6 [01:01<01:07, 22.61s/it]Querying SPARQL endpoint...:  67%|██████▋   | 4/6 [01:17<00:40, 20.15s/it]Querying SPARQL endpoint...:  83%|████████▎ | 5/6 [01:19<00:13, 13.37s/it]Querying SPARQL endpoint...: 100%|██████████| 6/6 [01:19<00:00,  8.95s/it]Querying SPARQL endpoint...: 100%|██████████| 6/6 [01:19<00:00, 13.27s/it]
2024-03-15 08:41:48,249 INFO: Splitting knowledge graph..
2024-03-15 08:41:53,324 INFO: Train set
2024-03-15 08:41:53,325 INFO: Number of entities: 284670
2024-03-15 08:41:53,325 INFO: Number of relation types: 7
2024-03-15 08:41:53,325 INFO: Number of triples: 447714 

2024-03-15 08:41:53,325 INFO: Test set
2024-03-15 08:41:53,325 INFO: Number of entities: 284670
2024-03-15 08:41:53,325 INFO: Number of relation types: 7
2024-03-15 08:41:53,325 INFO: Number of triples: 81676

2024-03-15 08:41:55,704 INFO: 2024-03-15 08:41:55.704487 - PARAMETERS
2024-03-15 08:41:55,704 INFO: 	 keywords : ['molecular-entity', 'phenotype', 'interaction', 'disease_plus_ortho', 'disease-ontology', 'phenotype-ontology']
2024-03-15 08:41:55,704 INFO: 	 method : ComplEx
2024-03-15 08:41:55,704 INFO: 	 dataset : celegans
2024-03-15 08:41:55,704 INFO: 	 query : None
2024-03-15 08:41:55,704 INFO: 	 normalize_parameters : True
2024-03-15 08:41:55,704 INFO: 	 train_classifier : ['rf', 'lr']
2024-03-15 08:41:55,704 INFO: 	 save_model : True
2024-03-15 08:41:55,704 INFO: 	 save_embeddings : True
2024-03-15 08:41:55,704 INFO: 	 n_epochs : 100
2024-03-15 08:41:55,704 INFO: 	 batch_size : 3072
2024-03-15 08:41:55,704 INFO: 	 lr : 0.0001
2024-03-15 08:41:55,705 INFO: 	 weight_decay : 0.0001
2024-03-15 08:41:55,705 INFO: 	 loss_fn : margin
2024-03-15 08:41:55,705 INFO: 	 ent_emb_dim : 50
2024-03-15 08:41:55,705 INFO: 	 eval_task : relation-prediction
2024-03-15 08:41:55,705 INFO: 	 split_ratio : 0.8
2024-03-15 08:41:55,705 INFO: 	 dissimilarity_type : L1
2024-03-15 08:41:55,705 INFO: 	 margin : 1.0
2024-03-15 08:41:55,705 INFO: 	 rel_emb_dim : 50
2024-03-15 08:41:55,705 INFO: 	 n_filters : 10
2024-03-15 08:41:55,705 INFO: 	 init_transe : False
2024-03-15 08:41:55,705 INFO: Training model ComplEx for 100 epochs...
2024-03-15 08:41:35.513241 - Query executed !
Function 'load_celegans' executed in 79.6071s
Function 'split' executed in 5.0746s
  0%|          | 0/100 [00:00<?, ?epoch/s]2024-03-15 08:41:56,555 INFO: 2024-03-15 08:41:56.555422 - ComplEx - Epoch 1 | mean loss: 3066.5337666550727, val loss: 3025.0297354239005
  1%|          | 1/100 [00:00<01:24,  1.18epoch/s]2024-03-15 08:41:57,394 INFO: 2024-03-15 08:41:57.394376 - ComplEx - Epoch 2 | mean loss: 3066.3598298373286, val loss: 3024.890190972222
  2%|▏         | 2/100 [00:01<01:22,  1.19epoch/s]2024-03-15 08:41:58,231 INFO: 2024-03-15 08:41:58.231381 - ComplEx - Epoch 3 | mean loss: 3065.2763036440497, val loss: 3024.1775987413193
  3%|▎         | 3/100 [00:02<01:21,  1.19epoch/s]2024-03-15 08:41:59,068 INFO: 2024-03-15 08:41:59.068504 - ComplEx - Epoch 4 | mean loss: 3061.841062780929, val loss: 3022.382826063368
  4%|▍         | 4/100 [00:03<01:20,  1.19epoch/s]2024-03-15 08:41:59,912 INFO: 2024-03-15 08:41:59.912018 - ComplEx - Epoch 5 | mean loss: 3054.835778641374, val loss: 3019.0518663194443
  5%|▌         | 5/100 [00:04<01:19,  1.19epoch/s]2024-03-15 08:42:00,765 INFO: 2024-03-15 08:42:00.765462 - ComplEx - Epoch 6 | mean loss: 3043.448516427654, val loss: 3013.925143771701
  6%|▌         | 6/100 [00:05<01:19,  1.18epoch/s]2024-03-15 08:42:01,602 INFO: 2024-03-15 08:42:01.601984 - ComplEx - Epoch 7 | mean loss: 3027.08099950503, val loss: 3006.662466543692
  7%|▋         | 7/100 [00:05<01:18,  1.19epoch/s]2024-03-15 08:42:02,441 INFO: 2024-03-15 08:42:02.441126 - ComplEx - Epoch 8 | mean loss: 3005.23686322774, val loss: 2997.1275453920716
  8%|▊         | 8/100 [00:06<01:17,  1.19epoch/s]2024-03-15 08:42:03,273 INFO: 2024-03-15 08:42:03.273163 - ComplEx - Epoch 9 | mean loss: 2977.5305543664385, val loss: 2984.9843840422454
  9%|▉         | 9/100 [00:07<01:16,  1.19epoch/s]2024-03-15 08:42:04,119 INFO: 2024-03-15 08:42:04.119590 - ComplEx - Epoch 10 | mean loss: 2943.566300901648, val loss: 2970.0931577329284
 10%|█         | 10/100 [00:08<01:15,  1.19epoch/s]2024-03-15 08:42:04,951 INFO: 2024-03-15 08:42:04.951063 - ComplEx - Epoch 11 | mean loss: 2903.04160591021, val loss: 2952.6580222800926
 11%|█         | 11/100 [00:09<01:14,  1.19epoch/s]2024-03-15 08:42:05,776 INFO: 2024-03-15 08:42:05.776065 - ComplEx - Epoch 12 | mean loss: 2855.6656443974744, val loss: 2931.950909649884
 12%|█▏        | 12/100 [00:10<01:13,  1.20epoch/s]2024-03-15 08:42:06,595 INFO: 2024-03-15 08:42:06.595353 - ComplEx - Epoch 13 | mean loss: 2801.227973833476, val loss: 2907.9938693576387
 13%|█▎        | 13/100 [00:10<01:12,  1.21epoch/s]2024-03-15 08:42:07,410 INFO: 2024-03-15 08:42:07.410746 - ComplEx - Epoch 14 | mean loss: 2739.4097064292596, val loss: 2881.117485894097
 14%|█▍        | 14/100 [00:11<01:10,  1.21epoch/s]2024-03-15 08:42:08,241 INFO: 2024-03-15 08:42:08.241547 - ComplEx - Epoch 15 | mean loss: 2670.0685299389984, val loss: 2850.7783248336227
 15%|█▌        | 15/100 [00:12<01:10,  1.21epoch/s]2024-03-15 08:42:09,078 INFO: 2024-03-15 08:42:09.078368 - ComplEx - Epoch 16 | mean loss: 2593.0988953472815, val loss: 2816.7923403139466
 16%|█▌        | 16/100 [00:13<01:09,  1.20epoch/s]2024-03-15 08:42:09,913 INFO: 2024-03-15 08:42:09.913518 - ComplEx - Epoch 17 | mean loss: 2508.3201302306293, val loss: 2778.674112955729
 17%|█▋        | 17/100 [00:14<01:09,  1.20epoch/s]2024-03-15 08:42:10,750 INFO: 2024-03-15 08:42:10.749927 - ComplEx - Epoch 18 | mean loss: 2415.293075770548, val loss: 2737.5642180266204
 18%|█▊        | 18/100 [00:15<01:08,  1.20epoch/s]2024-03-15 08:42:11,588 INFO: 2024-03-15 08:42:11.588519 - ComplEx - Epoch 19 | mean loss: 2314.0948544854987, val loss: 2692.5675817418983
 19%|█▉        | 19/100 [00:15<01:07,  1.20epoch/s]2024-03-15 08:42:12,425 INFO: 2024-03-15 08:42:12.424949 - ComplEx - Epoch 20 | mean loss: 2204.622446556614, val loss: 2642.886402271412
 20%|██        | 20/100 [00:16<01:06,  1.20epoch/s]2024-03-15 08:42:13,260 INFO: 2024-03-15 08:42:13.260806 - ComplEx - Epoch 21 | mean loss: 2086.609186877943, val loss: 2590.8823875144676
 21%|██        | 21/100 [00:17<01:05,  1.20epoch/s]2024-03-15 08:42:14,101 INFO: 2024-03-15 08:42:14.101332 - ComplEx - Epoch 22 | mean loss: 1959.7683556961686, val loss: 2534.1960087528937
 22%|██▏       | 22/100 [00:18<01:05,  1.19epoch/s]2024-03-15 08:42:14,956 INFO: 2024-03-15 08:42:14.956432 - ComplEx - Epoch 23 | mean loss: 1826.0956964362158, val loss: 2473.518215603299
 23%|██▎       | 23/100 [00:19<01:04,  1.19epoch/s]2024-03-15 08:42:15,805 INFO: 2024-03-15 08:42:15.805690 - ComplEx - Epoch 24 | mean loss: 1689.1035499050192, val loss: 2411.3092493127892
 24%|██▍       | 24/100 [00:20<01:04,  1.18epoch/s]2024-03-15 08:42:16,647 INFO: 2024-03-15 08:42:16.647765 - ComplEx - Epoch 25 | mean loss: 1553.241939596934, val loss: 2349.3924108434608
 25%|██▌       | 25/100 [00:20<01:03,  1.19epoch/s]2024-03-15 08:42:17,462 INFO: 2024-03-15 08:42:17.462809 - ComplEx - Epoch 26 | mean loss: 1423.5618369742615, val loss: 2288.1078468605324
 26%|██▌       | 26/100 [00:21<01:01,  1.20epoch/s]2024-03-15 08:42:18,269 INFO: 2024-03-15 08:42:18.269123 - ComplEx - Epoch 27 | mean loss: 1310.7767166764768, val loss: 2231.4719509548613
 27%|██▋       | 27/100 [00:22<01:00,  1.21epoch/s]2024-03-15 08:42:19,094 INFO: 2024-03-15 08:42:19.094422 - ComplEx - Epoch 28 | mean loss: 1211.5468555607208, val loss: 2182.765498408565
 28%|██▊       | 28/100 [00:23<00:59,  1.21epoch/s]2024-03-15 08:42:19,932 INFO: 2024-03-15 08:42:19.932106 - ComplEx - Epoch 29 | mean loss: 1125.6122438613684, val loss: 2136.3586380570023
 29%|██▉       | 29/100 [00:24<00:58,  1.21epoch/s]2024-03-15 08:42:20,776 INFO: 2024-03-15 08:42:20.776212 - ComplEx - Epoch 30 | mean loss: 1050.347139123368, val loss: 2094.001763237847
 30%|███       | 30/100 [00:25<00:58,  1.20epoch/s]2024-03-15 08:42:21,615 INFO: 2024-03-15 08:42:21.615434 - ComplEx - Epoch 31 | mean loss: 981.2980091669788, val loss: 2057.5918420862267
 31%|███       | 31/100 [00:25<00:57,  1.20epoch/s]2024-03-15 08:42:22,447 INFO: 2024-03-15 08:42:22.447509 - ComplEx - Epoch 32 | mean loss: 918.6777621752595, val loss: 2022.9197885018807
 32%|███▏      | 32/100 [00:26<00:56,  1.20epoch/s]2024-03-15 08:42:23,305 INFO: 2024-03-15 08:42:23.304977 - ComplEx - Epoch 33 | mean loss: 861.2929078193561, val loss: 1993.5810569480614
 33%|███▎      | 33/100 [00:27<00:56,  1.19epoch/s]2024-03-15 08:42:24,141 INFO: 2024-03-15 08:42:24.141809 - ComplEx - Epoch 34 | mean loss: 807.069855467914, val loss: 1968.7358308015046
 34%|███▍      | 34/100 [00:28<00:55,  1.19epoch/s]2024-03-15 08:42:24,967 INFO: 2024-03-15 08:42:24.967318 - ComplEx - Epoch 35 | mean loss: 756.0921429150725, val loss: 1942.499584056713
 35%|███▌      | 35/100 [00:29<00:54,  1.20epoch/s]2024-03-15 08:42:25,814 INFO: 2024-03-15 08:42:25.814794 - ComplEx - Epoch 36 | mean loss: 709.846442549196, val loss: 1921.4517754448784
 36%|███▌      | 36/100 [00:30<00:53,  1.19epoch/s]2024-03-15 08:42:26,648 INFO: 2024-03-15 08:42:26.648310 - ComplEx - Epoch 37 | mean loss: 666.5044245262668, val loss: 1899.9410083912037
 37%|███▋      | 37/100 [00:30<00:52,  1.19epoch/s]2024-03-15 08:42:27,489 INFO: 2024-03-15 08:42:27.489453 - ComplEx - Epoch 38 | mean loss: 626.1248120869676, val loss: 1880.9023098415798
 38%|███▊      | 38/100 [00:31<00:51,  1.19epoch/s]2024-03-15 08:42:28,331 INFO: 2024-03-15 08:42:28.330963 - ComplEx - Epoch 39 | mean loss: 593.7339872595383, val loss: 1865.4758165147568
 39%|███▉      | 39/100 [00:32<00:51,  1.19epoch/s]2024-03-15 08:42:29,167 INFO: 2024-03-15 08:42:29.167401 - ComplEx - Epoch 40 | mean loss: 566.3621018292152, val loss: 1848.4629403573495
 40%|████      | 40/100 [00:33<00:50,  1.19epoch/s]2024-03-15 08:42:30,003 INFO: 2024-03-15 08:42:30.002953 - ComplEx - Epoch 41 | mean loss: 541.8289798579804, val loss: 1834.454121907552
 41%|████      | 41/100 [00:34<00:49,  1.19epoch/s]2024-03-15 08:42:30,842 INFO: 2024-03-15 08:42:30.842056 - ComplEx - Epoch 42 | mean loss: 519.8712038536594, val loss: 1821.244845920139
 42%|████▏     | 42/100 [00:35<00:48,  1.19epoch/s]2024-03-15 08:42:31,678 INFO: 2024-03-15 08:42:31.677920 - ComplEx - Epoch 43 | mean loss: 500.61189682842934, val loss: 1807.7620465313946
 43%|████▎     | 43/100 [00:35<00:47,  1.19epoch/s]2024-03-15 08:42:32,513 INFO: 2024-03-15 08:42:32.513819 - ComplEx - Epoch 44 | mean loss: 481.9147878672979, val loss: 1796.8089011863426
 44%|████▍     | 44/100 [00:36<00:46,  1.19epoch/s]2024-03-15 08:42:33,345 INFO: 2024-03-15 08:42:33.345474 - ComplEx - Epoch 45 | mean loss: 464.55665081494476, val loss: 1782.5451049804688
 45%|████▌     | 45/100 [00:37<00:45,  1.20epoch/s]2024-03-15 08:42:34,206 INFO: 2024-03-15 08:42:34.206247 - ComplEx - Epoch 46 | mean loss: 447.65946155704864, val loss: 1770.0827749746818
 46%|████▌     | 46/100 [00:38<00:45,  1.19epoch/s]2024-03-15 08:42:35,076 INFO: 2024-03-15 08:42:35.076229 - ComplEx - Epoch 47 | mean loss: 432.00365495028564, val loss: 1760.2842226381656
 47%|████▋     | 47/100 [00:39<00:45,  1.17epoch/s]2024-03-15 08:42:35,912 INFO: 2024-03-15 08:42:35.912716 - ComplEx - Epoch 48 | mean loss: 417.1189752343583, val loss: 1748.7410368742767
 48%|████▊     | 48/100 [00:40<00:44,  1.18epoch/s]2024-03-15 08:42:36,747 INFO: 2024-03-15 08:42:36.747409 - ComplEx - Epoch 49 | mean loss: 403.59373495023544, val loss: 1739.142973723235
 49%|████▉     | 49/100 [00:41<00:42,  1.19epoch/s]2024-03-15 08:42:37,584 INFO: 2024-03-15 08:42:37.584511 - ComplEx - Epoch 50 | mean loss: 389.658946259381, val loss: 1729.1276425962094
 50%|█████     | 50/100 [00:41<00:42,  1.19epoch/s]2024-03-15 08:42:38,427 INFO: 2024-03-15 08:42:38.427106 - ComplEx - Epoch 51 | mean loss: 377.82560168227104, val loss: 1719.08221661603
 51%|█████     | 51/100 [00:42<00:41,  1.19epoch/s]2024-03-15 08:42:39,263 INFO: 2024-03-15 08:42:39.263356 - ComplEx - Epoch 52 | mean loss: 366.19869710321296, val loss: 1708.072245279948
 52%|█████▏    | 52/100 [00:43<00:40,  1.19epoch/s]2024-03-15 08:42:40,099 INFO: 2024-03-15 08:42:40.099147 - ComplEx - Epoch 53 | mean loss: 354.8505376005826, val loss: 1699.7714075159145
 53%|█████▎    | 53/100 [00:44<00:39,  1.19epoch/s]2024-03-15 08:42:40,933 INFO: 2024-03-15 08:42:40.933004 - ComplEx - Epoch 54 | mean loss: 344.4536596324346, val loss: 1690.7360229492188
 54%|█████▍    | 54/100 [00:45<00:38,  1.19epoch/s]2024-03-15 08:42:41,766 INFO: 2024-03-15 08:42:41.766602 - ComplEx - Epoch 55 | mean loss: 333.5224311776357, val loss: 1684.39374005353
 55%|█████▌    | 55/100 [00:46<00:37,  1.20epoch/s]2024-03-15 08:42:42,600 INFO: 2024-03-15 08:42:42.600813 - ComplEx - Epoch 56 | mean loss: 323.0964001694771, val loss: 1673.6752251519097
 56%|█████▌    | 56/100 [00:46<00:36,  1.20epoch/s]2024-03-15 08:42:43,432 INFO: 2024-03-15 08:42:43.431981 - ComplEx - Epoch 57 | mean loss: 313.0310396690891, val loss: 1667.9421092845776
 57%|█████▋    | 57/100 [00:47<00:35,  1.20epoch/s]2024-03-15 08:42:44,266 INFO: 2024-03-15 08:42:44.266905 - ComplEx - Epoch 58 | mean loss: 304.6781203648815, val loss: 1659.1599166304977
 58%|█████▊    | 58/100 [00:48<00:35,  1.20epoch/s]2024-03-15 08:42:45,106 INFO: 2024-03-15 08:42:45.106655 - ComplEx - Epoch 59 | mean loss: 294.96061314622017, val loss: 1652.4153532805267
 59%|█████▉    | 59/100 [00:49<00:34,  1.20epoch/s]2024-03-15 08:42:45,940 INFO: 2024-03-15 08:42:45.940549 - ComplEx - Epoch 60 | mean loss: 286.31545197473815, val loss: 1641.8737544307003
 60%|██████    | 60/100 [00:50<00:33,  1.20epoch/s]2024-03-15 08:42:46,774 INFO: 2024-03-15 08:42:46.774558 - ComplEx - Epoch 61 | mean loss: 278.57869683879693, val loss: 1637.6733963577835
 61%|██████    | 61/100 [00:51<00:32,  1.20epoch/s]2024-03-15 08:42:47,605 INFO: 2024-03-15 08:42:47.605736 - ComplEx - Epoch 62 | mean loss: 269.616702092837, val loss: 1627.5404889142071
 62%|██████▏   | 62/100 [00:51<00:31,  1.20epoch/s]2024-03-15 08:42:48,441 INFO: 2024-03-15 08:42:48.441836 - ComplEx - Epoch 63 | mean loss: 261.7375173699366, val loss: 1620.9923615632233
 63%|██████▎   | 63/100 [00:52<00:30,  1.20epoch/s]2024-03-15 08:42:49,277 INFO: 2024-03-15 08:42:49.277922 - ComplEx - Epoch 64 | mean loss: 253.26636970206482, val loss: 1612.6021547670719
 64%|██████▍   | 64/100 [00:53<00:30,  1.20epoch/s]2024-03-15 08:42:50,126 INFO: 2024-03-15 08:42:50.125990 - ComplEx - Epoch 65 | mean loss: 246.04374731403507, val loss: 1607.6780146846065
 65%|██████▌   | 65/100 [00:54<00:29,  1.19epoch/s]2024-03-15 08:42:50,960 INFO: 2024-03-15 08:42:50.960807 - ComplEx - Epoch 66 | mean loss: 238.47558946478856, val loss: 1598.4081115722656
 66%|██████▌   | 66/100 [00:55<00:28,  1.19epoch/s]2024-03-15 08:42:51,792 INFO: 2024-03-15 08:42:51.792036 - ComplEx - Epoch 67 | mean loss: 230.8678824150399, val loss: 1592.099674931279
 67%|██████▋   | 67/100 [00:56<00:27,  1.20epoch/s]2024-03-15 08:42:52,629 INFO: 2024-03-15 08:42:52.629916 - ComplEx - Epoch 68 | mean loss: 224.072546828283, val loss: 1584.0616669831452
 68%|██████▊   | 68/100 [00:56<00:26,  1.20epoch/s]2024-03-15 08:42:53,461 INFO: 2024-03-15 08:42:53.461608 - ComplEx - Epoch 69 | mean loss: 217.43909804461754, val loss: 1580.6812811957466
 69%|██████▉   | 69/100 [00:57<00:25,  1.20epoch/s]2024-03-15 08:42:54,297 INFO: 2024-03-15 08:42:54.297192 - ComplEx - Epoch 70 | mean loss: 211.0353884892921, val loss: 1572.7083423755787
 70%|███████   | 70/100 [00:58<00:25,  1.20epoch/s]2024-03-15 08:42:55,132 INFO: 2024-03-15 08:42:55.132455 - ComplEx - Epoch 71 | mean loss: 204.35766077694828, val loss: 1570.0457243742767
 71%|███████   | 71/100 [00:59<00:24,  1.20epoch/s]2024-03-15 08:42:55,974 INFO: 2024-03-15 08:42:55.974296 - ComplEx - Epoch 72 | mean loss: 198.7950597135988, val loss: 1558.3688874421296
 72%|███████▏  | 72/100 [01:00<00:23,  1.19epoch/s]2024-03-15 08:42:56,812 INFO: 2024-03-15 08:42:56.812418 - ComplEx - Epoch 73 | mean loss: 192.57545656700657, val loss: 1552.0117820457176
 73%|███████▎  | 73/100 [01:01<00:22,  1.19epoch/s]2024-03-15 08:42:57,632 INFO: 2024-03-15 08:42:57.632293 - ComplEx - Epoch 74 | mean loss: 186.0291803177089, val loss: 1547.9988425925926
 74%|███████▍  | 74/100 [01:01<00:21,  1.20epoch/s]2024-03-15 08:42:58,470 INFO: 2024-03-15 08:42:58.470797 - ComplEx - Epoch 75 | mean loss: 180.7515840661036, val loss: 1543.2714742024739
 75%|███████▌  | 75/100 [01:02<00:20,  1.20epoch/s]2024-03-15 08:42:59,320 INFO: 2024-03-15 08:42:59.320185 - ComplEx - Epoch 76 | mean loss: 174.92203619708755, val loss: 1535.645311143663
 76%|███████▌  | 76/100 [01:03<00:20,  1.19epoch/s]2024-03-15 08:43:00,150 INFO: 2024-03-15 08:43:00.150613 - ComplEx - Epoch 77 | mean loss: 169.62708650876397, val loss: 1532.0740299931279
 77%|███████▋  | 77/100 [01:04<00:19,  1.20epoch/s]2024-03-15 08:43:00,995 INFO: 2024-03-15 08:43:00.995557 - ComplEx - Epoch 78 | mean loss: 164.57947184288338, val loss: 1523.66700914171
 78%|███████▊  | 78/100 [01:05<00:18,  1.19epoch/s]2024-03-15 08:43:01,829 INFO: 2024-03-15 08:43:01.829307 - ComplEx - Epoch 79 | mean loss: 159.1182701555017, val loss: 1518.9608391655815
 79%|███████▉  | 79/100 [01:06<00:17,  1.19epoch/s]2024-03-15 08:43:02,686 INFO: 2024-03-15 08:43:02.685987 - ComplEx - Epoch 80 | mean loss: 154.40041510699547, val loss: 1517.3271947790074
 80%|████████  | 80/100 [01:06<00:16,  1.19epoch/s]2024-03-15 08:43:03,526 INFO: 2024-03-15 08:43:03.526865 - ComplEx - Epoch 81 | mean loss: 149.34438244937218, val loss: 1509.1169591833043
 81%|████████  | 81/100 [01:07<00:16,  1.19epoch/s]2024-03-15 08:43:04,363 INFO: 2024-03-15 08:43:04.363430 - ComplEx - Epoch 82 | mean loss: 144.45467878367802, val loss: 1500.923294632523
 82%|████████▏ | 82/100 [01:08<00:15,  1.19epoch/s]2024-03-15 08:43:05,199 INFO: 2024-03-15 08:43:05.199677 - ComplEx - Epoch 83 | mean loss: 140.0204206166202, val loss: 1500.176341869213
 83%|████████▎ | 83/100 [01:09<00:14,  1.19epoch/s]2024-03-15 08:43:06,038 INFO: 2024-03-15 08:43:06.038213 - ComplEx - Epoch 84 | mean loss: 135.55710812790753, val loss: 1493.2022727683739
 84%|████████▍ | 84/100 [01:10<00:13,  1.19epoch/s]2024-03-15 08:43:06,867 INFO: 2024-03-15 08:43:06.867918 - ComplEx - Epoch 85 | mean loss: 130.2583154717537, val loss: 1488.4604390462239
 85%|████████▌ | 85/100 [01:11<00:12,  1.20epoch/s]2024-03-15 08:43:07,701 INFO: 2024-03-15 08:43:07.701858 - ComplEx - Epoch 86 | mean loss: 126.14412891701477, val loss: 1484.8808209454571
 86%|████████▌ | 86/100 [01:11<00:11,  1.20epoch/s]2024-03-15 08:43:08,559 INFO: 2024-03-15 08:43:08.559827 - ComplEx - Epoch 87 | mean loss: 121.92723647209063, val loss: 1478.7231287073207
 87%|████████▋ | 87/100 [01:12<00:10,  1.19epoch/s]2024-03-15 08:43:09,390 INFO: 2024-03-15 08:43:09.390781 - ComplEx - Epoch 88 | mean loss: 116.92976918939041, val loss: 1476.3719618055557
 88%|████████▊ | 88/100 [01:13<00:10,  1.19epoch/s]2024-03-15 08:43:10,214 INFO: 2024-03-15 08:43:10.214668 - ComplEx - Epoch 89 | mean loss: 112.6880567890324, val loss: 1468.0806749131943
 89%|████████▉ | 89/100 [01:14<00:09,  1.20epoch/s]2024-03-15 08:43:11,046 INFO: 2024-03-15 08:43:11.046105 - ComplEx - Epoch 90 | mean loss: 108.01432900232811, val loss: 1462.4963096336082
 90%|█████████ | 90/100 [01:15<00:08,  1.20epoch/s]2024-03-15 08:43:11,879 INFO: 2024-03-15 08:43:11.879332 - ComplEx - Epoch 91 | mean loss: 104.22212963234888, val loss: 1460.3755289713542
 91%|█████████ | 91/100 [01:16<00:07,  1.20epoch/s]2024-03-15 08:43:12,686 INFO: 2024-03-15 08:43:12.686300 - ComplEx - Epoch 92 | mean loss: 99.3542676037305, val loss: 1454.5168502242477
 92%|█████████▏| 92/100 [01:16<00:06,  1.21epoch/s]2024-03-15 08:43:13,521 INFO: 2024-03-15 08:43:13.521104 - ComplEx - Epoch 93 | mean loss: 95.19156443582823, val loss: 1452.3713164152923
 93%|█████████▎| 93/100 [01:17<00:05,  1.21epoch/s]2024-03-15 08:43:14,353 INFO: 2024-03-15 08:43:14.353308 - ComplEx - Epoch 94 | mean loss: 89.99776636737667, val loss: 1444.7549257631656
 94%|█████████▍| 94/100 [01:18<00:04,  1.21epoch/s]2024-03-15 08:43:15,188 INFO: 2024-03-15 08:43:15.188300 - ComplEx - Epoch 95 | mean loss: 86.33997556934618, val loss: 1442.3269981101707
 95%|█████████▌| 95/100 [01:19<00:04,  1.20epoch/s]2024-03-15 08:43:16,019 INFO: 2024-03-15 08:43:16.019676 - ComplEx - Epoch 96 | mean loss: 82.00586673658188, val loss: 1439.4908481174045
 96%|█████████▌| 96/100 [01:20<00:03,  1.20epoch/s]2024-03-15 08:43:16,859 INFO: 2024-03-15 08:43:16.859402 - ComplEx - Epoch 97 | mean loss: 77.55656582362032, val loss: 1431.9089886700665
 97%|█████████▋| 97/100 [01:21<00:02,  1.20epoch/s]2024-03-15 08:43:17,692 INFO: 2024-03-15 08:43:17.692355 - ComplEx - Epoch 98 | mean loss: 73.28754303879934, val loss: 1427.9789202654804
 98%|█████████▊| 98/100 [01:21<00:01,  1.20epoch/s]2024-03-15 08:43:18,525 INFO: 2024-03-15 08:43:18.525801 - ComplEx - Epoch 99 | mean loss: 69.12680330015209, val loss: 1425.4054768880208
 99%|█████████▉| 99/100 [01:22<00:00,  1.20epoch/s]2024-03-15 08:43:19,358 INFO: 2024-03-15 08:43:19.358852 - ComplEx - Epoch 100 | mean loss: 65.2544890168595, val loss: 1418.3143118399162
100%|██████████| 100/100 [01:23<00:00,  1.20epoch/s]100%|██████████| 100/100 [01:23<00:00,  1.20epoch/s]
2024-03-15 08:43:19,359 INFO: 2024-03-15 08:43:19.359291 - Finished Training of ComplEx !

2024-03-15 08:43:21,138 INFO: 2024-03-15 08:43:21.138361 - Evaluating..
Relation prediction evaluation:   0%|          | 0/310 [00:00<?, ?batch/s]Relation prediction evaluation:   4%|▎         | 11/310 [00:00<00:02, 109.66batch/s]Relation prediction evaluation:   7%|▋         | 23/310 [00:00<00:02, 111.15batch/s]Relation prediction evaluation:  11%|█▏        | 35/310 [00:00<00:02, 112.50batch/s]Relation prediction evaluation:  15%|█▌        | 47/310 [00:00<00:02, 113.18batch/s]Relation prediction evaluation:  19%|█▉        | 59/310 [00:00<00:02, 113.51batch/s]Relation prediction evaluation:  23%|██▎       | 71/310 [00:00<00:02, 113.78batch/s]Relation prediction evaluation:  27%|██▋       | 83/310 [00:00<00:01, 113.93batch/s]Relation prediction evaluation:  31%|███       | 95/310 [00:00<00:01, 114.06batch/s]Relation prediction evaluation:  35%|███▍      | 107/310 [00:00<00:01, 114.10batch/s]Relation prediction evaluation:  38%|███▊      | 119/310 [00:01<00:01, 114.09batch/s]Relation prediction evaluation:  42%|████▏     | 131/310 [00:01<00:01, 114.13batch/s]Relation prediction evaluation:  46%|████▌     | 143/310 [00:01<00:01, 114.00batch/s]Relation prediction evaluation:  50%|█████     | 155/310 [00:01<00:01, 114.06batch/s]Relation prediction evaluation:  54%|█████▍    | 167/310 [00:01<00:01, 114.06batch/s]Relation prediction evaluation:  58%|█████▊    | 179/310 [00:01<00:01, 114.17batch/s]Relation prediction evaluation:  62%|██████▏   | 191/310 [00:01<00:01, 114.34batch/s]Relation prediction evaluation:  65%|██████▌   | 203/310 [00:01<00:00, 114.44batch/s]Relation prediction evaluation:  69%|██████▉   | 215/310 [00:01<00:00, 114.52batch/s]Relation prediction evaluation:  73%|███████▎  | 227/310 [00:01<00:00, 114.59batch/s]Relation prediction evaluation:  77%|███████▋  | 239/310 [00:02<00:00, 114.64batch/s]Relation prediction evaluation:  81%|████████  | 251/310 [00:02<00:00, 114.72batch/s]Relation prediction evaluation:  85%|████████▍ | 263/310 [00:02<00:00, 114.74batch/s]Relation prediction evaluation:  89%|████████▊ | 275/310 [00:02<00:00, 114.57batch/s]Relation prediction evaluation:  93%|█████████▎| 287/310 [00:02<00:00, 114.38batch/s]Relation prediction evaluation:  96%|█████████▋| 299/310 [00:02<00:00, 114.22batch/s]Relation prediction evaluation: 100%|██████████| 310/310 [00:02<00:00, 114.26batch/s]
2024-03-15 08:43:23,853 INFO: {'http://semanticscience.org/resource/SIO_000628': 0, 'http://semanticscience.org/resource/SIO_001279': 1, 'http://www.semanticweb.org/needed-terms#001': 2, 'http://www.semanticweb.org/needed-terms#009': 3, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 4, 'http://www.w3.org/2000/01/rdf-schema#label': 5, 'http://www.w3.org/2000/01/rdf-schema#subClassOf': 6}
2024-03-15 08:43:23,884 INFO: 
 [[    0     0     0     0     0     0     0     0]
 [ 8881 13331 37798  3211  1646     0  1451     0]
 [  272  1234  2571   937   293   966   412     0]
 [  164   443   650   575   174   482   272     0]
 [  167   365   405   465   116   294   200     0]
 [  118   266   272   407    90   231   204     0]
 [   95   191   192   298    69   184   135     0]
 [   86   239   160   250    54   216   144     0]]
2024-03-15 08:43:23,884 INFO: 2024-03-15 08:43:23.884393 - EMBEDDING MODEL EVALUATION RESULTS:
2024-03-15 08:43:23,884 INFO: Task : relation-prediction
2024-03-15 08:43:23,885 INFO: Hit@1 : 0.8119643330574036
2024-03-15 08:43:23,886 INFO: Hit@3 : 0.9276041984558105
2024-03-15 08:43:23,886 INFO: Hit@5 : 0.971680760383606
2024-03-15 08:43:23,887 INFO: Hit@10 : 1.0
2024-03-15 08:43:23,887 INFO: Mean Rank : 1.4567681550979614
2024-03-15 08:43:23,888 INFO: MRR : 0.8785842061042786
2024-03-15 08:43:23,896 INFO: Training of Embedding Model done !

2024-03-15 08:43:23,904 INFO: Converting test set to embeddings...
Function 'evaluate_emb_model' executed in 2.7505s
Function 'train' executed in 108.3835s
Generating embeddings for dataset:   0%|          | 0/160 [00:00<?, ?it/s]Generating embeddings for dataset:  10%|█         | 16/160 [00:00<00:00, 156.56it/s]Generating embeddings for dataset:  20%|██        | 32/160 [00:00<00:00, 157.04it/s]Generating embeddings for dataset:  30%|███       | 48/160 [00:00<00:00, 156.99it/s]Generating embeddings for dataset:  40%|████      | 64/160 [00:00<00:00, 157.10it/s]Generating embeddings for dataset:  50%|█████     | 80/160 [00:00<00:00, 156.04it/s]Generating embeddings for dataset:  60%|██████    | 96/160 [00:00<00:00, 155.02it/s]Generating embeddings for dataset:  70%|███████   | 112/160 [00:00<00:00, 154.24it/s]Generating embeddings for dataset:  80%|████████  | 128/160 [00:00<00:00, 153.64it/s]Generating embeddings for dataset:  90%|█████████ | 144/160 [00:00<00:00, 152.96it/s]Generating embeddings for dataset: 100%|██████████| 160/160 [00:01<00:00, 153.57it/s]Generating embeddings for dataset: 100%|██████████| 160/160 [00:01<00:00, 154.63it/s]
2024-03-15 08:43:45,180 INFO: Test set converted. It will be used to train the classifier

2024-03-15 08:43:45,181 INFO: Training classifier...
2024-03-15 08:43:45,181 INFO: Model types: ['rf', 'lr']
2024-03-15 08:43:45,273 INFO: Experiment Setup:
2024-03-15 08:43:45,275 INFO: PyCaret ClassificationExperiment
2024-03-15 08:43:45,275 INFO: Logging name: clf-default-name
2024-03-15 08:43:45,275 INFO: ML Usecase: MLUsecase.CLASSIFICATION
2024-03-15 08:43:45,275 INFO: version 3.2.0
2024-03-15 08:43:45,275 INFO: Initializing setup()
2024-03-15 08:43:45,275 INFO: self.USI: 18fb
2024-03-15 08:43:45,275 INFO: self._variable_keys: {'is_multiclass', 'fold_shuffle_param', 'log_plots_param', 'data', 'fix_imbalance', 'gpu_n_jobs_param', 'y_train', 'exp_name_log', 'logging_param', 'fold_generator', 'y', 'idx', 'fold_groups_param', 'pipeline', 'memory', 'exp_id', '_ml_usecase', 'X_test', 'USI', 'n_jobs_param', 'seed', 'X_train', 'X', 'html_param', 'target_param', 'gpu_param', 'y_test', '_available_plots'}
2024-03-15 08:43:45,275 INFO: Checking environment
2024-03-15 08:43:45,275 INFO: python_version: 3.10.13
2024-03-15 08:43:45,275 INFO: python_build: ('main', 'Sep 11 2023 13:44:35')
2024-03-15 08:43:45,275 INFO: machine: x86_64
2024-03-15 08:43:45,277 INFO: platform: Linux-6.1.0-18-amd64-x86_64-with-glibc2.36
2024-03-15 08:43:45,278 INFO: Memory: svmem(total=270124969984, available=249209864192, percent=7.7, used=18676264960, free=23859519488, active=75917053952, inactive=160585441280, buffers=3597434880, cached=223991750656, shared=333881344, slab=7817924608)
2024-03-15 08:43:45,279 INFO: Physical Core: 32
2024-03-15 08:43:45,279 INFO: Logical Core: 64
2024-03-15 08:43:45,279 INFO: Checking libraries
2024-03-15 08:43:45,279 INFO: System:
2024-03-15 08:43:45,279 INFO:     python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]
2024-03-15 08:43:45,279 INFO: executable: /home/heligon/anaconda3/envs/Antoine/bin/python
2024-03-15 08:43:45,279 INFO:    machine: Linux-6.1.0-18-amd64-x86_64-with-glibc2.36
2024-03-15 08:43:45,279 INFO: PyCaret required dependencies:
2024-03-15 08:43:45,300 INFO:                  pip: 23.3.1
2024-03-15 08:43:45,300 INFO:           setuptools: 68.2.2
2024-03-15 08:43:45,300 INFO:              pycaret: 3.2.0
2024-03-15 08:43:45,300 INFO:              IPython: 8.21.0
2024-03-15 08:43:45,300 INFO:           ipywidgets: 8.1.1
2024-03-15 08:43:45,300 INFO:                 tqdm: 4.66.1
2024-03-15 08:43:45,300 INFO:                numpy: 1.25.2
2024-03-15 08:43:45,300 INFO:               pandas: 1.5.3
2024-03-15 08:43:45,300 INFO:               jinja2: 3.1.3
2024-03-15 08:43:45,300 INFO:                scipy: 1.10.1
2024-03-15 08:43:45,300 INFO:               joblib: 1.3.2
2024-03-15 08:43:45,300 INFO:              sklearn: 1.2.2
2024-03-15 08:43:45,300 INFO:                 pyod: 1.1.2
2024-03-15 08:43:45,300 INFO:             imblearn: 0.12.0
2024-03-15 08:43:45,300 INFO:    category_encoders: 2.6.3
2024-03-15 08:43:45,300 INFO:             lightgbm: 4.3.0
2024-03-15 08:43:45,300 INFO:                numba: 0.59.0
2024-03-15 08:43:45,300 INFO:             requests: 2.31.0
2024-03-15 08:43:45,300 INFO:           matplotlib: 3.6.0
2024-03-15 08:43:45,300 INFO:           scikitplot: 0.3.7
2024-03-15 08:43:45,300 INFO:          yellowbrick: 1.5
2024-03-15 08:43:45,300 INFO:               plotly: 5.18.0
2024-03-15 08:43:45,300 INFO:     plotly-resampler: Not installed
2024-03-15 08:43:45,300 INFO:              kaleido: 0.2.1
2024-03-15 08:43:45,300 INFO:            schemdraw: 0.15
2024-03-15 08:43:45,300 INFO:          statsmodels: 0.14.1
2024-03-15 08:43:45,300 INFO:               sktime: 0.21.1
2024-03-15 08:43:45,300 INFO:                tbats: 1.1.3
2024-03-15 08:43:45,300 INFO:             pmdarima: 2.0.4
2024-03-15 08:43:45,300 INFO:               psutil: 5.9.8
2024-03-15 08:43:45,301 INFO:           markupsafe: 2.1.5
2024-03-15 08:43:45,301 INFO:              pickle5: Not installed
2024-03-15 08:43:45,301 INFO:          cloudpickle: 3.0.0
2024-03-15 08:43:45,301 INFO:          deprecation: 2.1.0
2024-03-15 08:43:45,301 INFO:               xxhash: 3.4.1
2024-03-15 08:43:45,301 INFO:            wurlitzer: 3.0.3
2024-03-15 08:43:45,301 INFO: PyCaret optional dependencies:
2024-03-15 08:43:45,324 INFO:                 shap: Not installed
2024-03-15 08:43:45,324 INFO:            interpret: Not installed
2024-03-15 08:43:45,324 INFO:                 umap: Not installed
2024-03-15 08:43:45,325 INFO:      ydata_profiling: Not installed
2024-03-15 08:43:45,325 INFO:   explainerdashboard: Not installed
2024-03-15 08:43:45,325 INFO:              autoviz: Not installed
2024-03-15 08:43:45,325 INFO:            fairlearn: Not installed
2024-03-15 08:43:45,325 INFO:           deepchecks: Not installed
2024-03-15 08:43:45,325 INFO:              xgboost: Not installed
2024-03-15 08:43:45,325 INFO:             catboost: Not installed
2024-03-15 08:43:45,325 INFO:               kmodes: Not installed
2024-03-15 08:43:45,325 INFO:              mlxtend: Not installed
2024-03-15 08:43:45,325 INFO:        statsforecast: Not installed
2024-03-15 08:43:45,325 INFO:         tune_sklearn: Not installed
2024-03-15 08:43:45,325 INFO:                  ray: Not installed
2024-03-15 08:43:45,325 INFO:             hyperopt: Not installed
2024-03-15 08:43:45,325 INFO:               optuna: Not installed
2024-03-15 08:43:45,325 INFO:                skopt: Not installed
2024-03-15 08:43:45,325 INFO:               mlflow: Not installed
2024-03-15 08:43:45,325 INFO:               gradio: Not installed
2024-03-15 08:43:45,325 INFO:              fastapi: Not installed
2024-03-15 08:43:45,325 INFO:              uvicorn: Not installed
2024-03-15 08:43:45,325 INFO:               m2cgen: Not installed
2024-03-15 08:43:45,325 INFO:            evidently: Not installed
2024-03-15 08:43:45,325 INFO:                fugue: Not installed
2024-03-15 08:43:45,325 INFO:            streamlit: Not installed
2024-03-15 08:43:45,325 INFO:              prophet: Not installed
2024-03-15 08:43:45,325 INFO: None
2024-03-15 08:43:45,325 INFO: Set up data.
2024-03-15 08:43:45,622 INFO: Set up folding strategy.
2024-03-15 08:43:45,622 INFO: Set up train/test split.
2024-03-15 08:43:45,987 INFO: Set up index.
2024-03-15 08:43:45,990 INFO: Assigning column types.
2024-03-15 08:43:46,271 INFO: Engine successfully changes for model 'lr' to 'sklearn'.
2024-03-15 08:43:46,303 INFO: Engine for model 'knn' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,305 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,360 INFO: Engine for model 'knn' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,360 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,380 INFO: Engine successfully changes for model 'knn' to 'sklearn'.
2024-03-15 08:43:46,413 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,465 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,486 INFO: Engine successfully changes for model 'rbfsvm' to 'sklearn'.
2024-03-15 08:43:46,592 INFO: Preparing preprocessing pipeline...
2024-03-15 08:43:46,635 INFO: Set up simple imputation.
2024-03-15 08:43:47,414 INFO: Finished creating preprocessing pipeline.
2024-03-15 08:43:47,418 INFO: Pipeline: Pipeline(memory=FastMemory(location=/tmp/joblib),
         steps=[('numerical_imputer',
                 TransformerWrapper(exclude=None,
                                    include=['0', '1', '2', '3', '4', '5', '6',
                                             '7', '8', '9', '10', '11', '12',
                                             '13', '14', '15', '16', '17', '18',
                                             '19', '20', '21', '22', '23', '24',
                                             '25', '26', '27', '28', '29', ...],
                                    transformer=SimpleImputer(add_indicator=False,
                                                              copy=True,
                                                              fill_value=None,
                                                              keep_empty_features=False,
                                                              missing_values=nan,
                                                              strategy='mean',
                                                              verbose='deprecated'))),
                ('categorical_imputer',
                 TransformerWrapper(exclude=None, include=[],
                                    transformer=SimpleImputer(add_indicator=False,
                                                              copy=True,
                                                              fill_value=None,
                                                              keep_empty_features=False,
                                                              missing_values=nan,
                                                              strategy='most_frequent',
                                                              verbose='deprecated')))],
         verbose=False)
2024-03-15 08:43:47,418 INFO: Creating final display dataframe.
2024-03-15 08:43:48,547 INFO: Setup _display_container:                     Description             Value
0                    Session id              5353
1                        Target              link
2                   Target type            Binary
3           Original data shape     (163352, 201)
4        Transformed data shape     (163352, 201)
5   Transformed train set shape     (130681, 201)
6    Transformed test set shape      (32671, 201)
7              Numeric features               200
8                    Preprocess              True
9               Imputation type            simple
10           Numeric imputation              mean
11       Categorical imputation              mode
12               Fold Generator   StratifiedKFold
13                  Fold Number                10
14                     CPU Jobs                -1
15                      Use GPU             False
16               Log Experiment             False
17              Experiment Name  clf-default-name
18                          USI              18fb
2024-03-15 08:43:48,655 INFO: setup() successfully completed in 3.38s...............
2024-03-15 08:43:48,656 INFO: MODEL - rf
2024-03-15 08:43:48,656 INFO: Initializing create_model()
2024-03-15 08:43:48,656 INFO: create_model(self=<pycaret.classification.oop.ClassificationExperiment object at 0x7fdea3d037f0>, estimator=rf, fold=None, round=4, cross_validation=True, predict=True, fit_kwargs=None, groups=None, refit=True, probability_threshold=None, experiment_custom_tags=None, verbose=False, system=True, add_to_model_list=True, metrics=None, display=None, model_only=True, return_train_score=False, error_score=0.0, kwargs={})
2024-03-15 08:43:48,656 INFO: Checking exceptions
2024-03-15 08:43:48,657 INFO: Importing libraries
2024-03-15 08:43:48,657 INFO: Copying training dataset
2024-03-15 08:43:48,971 INFO: Defining folds
2024-03-15 08:43:48,972 INFO: Declaring metric variables
2024-03-15 08:43:48,972 INFO: Importing untrained model
2024-03-15 08:43:48,972 INFO: Random Forest Classifier Imported successfully
2024-03-15 08:43:48,972 INFO: Starting cross validation
2024-03-15 08:43:48,973 INFO: Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
2024-03-15 08:45:12,459 INFO: Calculating mean and std
2024-03-15 08:45:12,460 INFO: Creating metrics dataframe
2024-03-15 08:45:12,466 INFO: Finalizing model
2024-03-15 08:45:24,115 INFO: Uploading results into container
2024-03-15 08:45:24,117 INFO: Uploading model into container now
2024-03-15 08:45:24,118 INFO: _master_model_container: 1
2024-03-15 08:45:24,118 INFO: _display_container: 2
2024-03-15 08:45:24,119 INFO: RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=5353, verbose=0, warm_start=False)
2024-03-15 08:45:24,119 INFO: create_model() successfully completed......................................
2024-03-15 08:45:24,773 INFO: CONFIG -
 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=5353, verbose=0, warm_start=False)
2024-03-15 08:45:24,780 INFO: RESULTS -
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold                                                          
0       0.9410  0.9815  0.9734  0.9142  0.9429  0.8820  0.8839
1       0.9394  0.9810  0.9737  0.9112  0.9414  0.8788  0.8809
2       0.9430  0.9813  0.9729  0.9180  0.9446  0.8860  0.8876
3       0.9372  0.9804  0.9743  0.9070  0.9394  0.8743  0.8768
4       0.9420  0.9813  0.9769  0.9132  0.9440  0.8840  0.8862
5       0.9377  0.9809  0.9737  0.9083  0.9399  0.8754  0.8777
6       0.9415  0.9825  0.9760  0.9130  0.9434  0.8829  0.8850
7       0.9429  0.9814  0.9752  0.9160  0.9447  0.8858  0.8877
8       0.9409  0.9820  0.9735  0.9139  0.9428  0.8818  0.8837
9       0.9445  0.9818  0.9784  0.9163  0.9463  0.8890  0.8911
Mean    0.9410  0.9814  0.9748  0.9131  0.9429  0.8820  0.8840
Std     0.0022  0.0005  0.0017  0.0033  0.0021  0.0044  0.0043
2024-03-15 08:45:25,359 INFO: MODEL - lr
2024-03-15 08:45:25,359 INFO: Initializing create_model()
2024-03-15 08:45:25,359 INFO: create_model(self=<pycaret.classification.oop.ClassificationExperiment object at 0x7fdea3d037f0>, estimator=lr, fold=None, round=4, cross_validation=True, predict=True, fit_kwargs=None, groups=None, refit=True, probability_threshold=None, experiment_custom_tags=None, verbose=False, system=True, add_to_model_list=True, metrics=None, display=None, model_only=True, return_train_score=False, error_score=0.0, kwargs={})
2024-03-15 08:45:25,359 INFO: Checking exceptions
2024-03-15 08:45:25,360 INFO: Importing libraries
2024-03-15 08:45:25,360 INFO: Copying training dataset
2024-03-15 08:45:25,685 INFO: Defining folds
2024-03-15 08:45:25,685 INFO: Declaring metric variables
2024-03-15 08:45:25,686 INFO: Importing untrained model
2024-03-15 08:45:25,686 INFO: Logistic Regression Imported successfully
2024-03-15 08:45:25,686 INFO: Starting cross validation
2024-03-15 08:45:25,687 INFO: Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
2024-03-15 08:45:33,399 INFO: Calculating mean and std
2024-03-15 08:45:33,400 INFO: Creating metrics dataframe
2024-03-15 08:45:33,405 INFO: Finalizing model
2024-03-15 08:45:36,042 INFO: Uploading results into container
2024-03-15 08:45:36,043 INFO: Uploading model into container now
2024-03-15 08:45:36,044 INFO: _master_model_container: 2
2024-03-15 08:45:36,044 INFO: _display_container: 3
2024-03-15 08:45:36,044 INFO: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5353, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
2024-03-15 08:45:36,044 INFO: create_model() successfully completed......................................
2024-03-15 08:45:36,703 INFO: CONFIG -
 LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5353, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
2024-03-15 08:45:36,710 INFO: RESULTS -
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold                                                          
0       0.8995  0.9442  0.9273  0.8785  0.9022  0.7991  0.8003
1       0.8972  0.9454  0.9256  0.8758  0.9000  0.7943  0.7956
2       0.9025  0.9489  0.9265  0.8841  0.9048  0.8050  0.8060
3       0.9013  0.9442  0.9328  0.8775  0.9043  0.8026  0.8042
4       0.9014  0.9443  0.9256  0.8829  0.9038  0.8029  0.8038
5       0.8990  0.9439  0.9270  0.8778  0.9017  0.7980  0.7992
6       0.9005  0.9462  0.9264  0.8808  0.9030  0.8010  0.8021
7       0.9032  0.9482  0.9291  0.8833  0.9056  0.8064  0.8075
8       0.9011  0.9463  0.9264  0.8818  0.9036  0.8023  0.8033
9       0.9058  0.9504  0.9302  0.8869  0.9080  0.8116  0.8126
Mean    0.9012  0.9462  0.9277  0.8809  0.9037  0.8023  0.8035
Std     0.0023  0.0022  0.0022  0.0033  0.0021  0.0045  0.0044
2024-03-15 08:45:37,275 INFO: Classifier trained !

2024-03-15 08:45:37,276 INFO: Saving embeddings...
Function 'generate' executed in 21.2764s
Transformation Pipeline and Model Successfully Saved
Transformation Pipeline and Model Successfully Saved
Generating embeddings for dataset:   0%|          | 0/875 [00:00<?, ?it/s]Generating embeddings for dataset:   2%|▏         | 16/875 [00:00<00:05, 155.21it/s]Generating embeddings for dataset:   4%|▎         | 32/875 [00:00<00:05, 155.97it/s]Generating embeddings for dataset:   5%|▌         | 48/875 [00:00<00:05, 155.45it/s]Generating embeddings for dataset:   7%|▋         | 64/875 [00:00<00:05, 155.87it/s]Generating embeddings for dataset:   9%|▉         | 80/875 [00:00<00:05, 155.96it/s]Generating embeddings for dataset:  11%|█         | 96/875 [00:00<00:04, 155.93it/s]Generating embeddings for dataset:  13%|█▎        | 112/875 [00:00<00:04, 155.77it/s]Generating embeddings for dataset:  15%|█▍        | 128/875 [00:00<00:04, 155.16it/s]Generating embeddings for dataset:  16%|█▋        | 144/875 [00:00<00:04, 154.61it/s]Generating embeddings for dataset:  18%|█▊        | 160/875 [00:01<00:04, 153.62it/s]Generating embeddings for dataset:  20%|██        | 176/875 [00:01<00:04, 152.31it/s]Generating embeddings for dataset:  22%|██▏       | 192/875 [00:01<00:04, 151.18it/s]Generating embeddings for dataset:  24%|██▍       | 208/875 [00:01<00:04, 150.51it/s]Generating embeddings for dataset:  26%|██▌       | 224/875 [00:01<00:04, 149.22it/s]Generating embeddings for dataset:  27%|██▋       | 239/875 [00:01<00:04, 148.90it/s]Generating embeddings for dataset:  29%|██▉       | 254/875 [00:01<00:04, 148.43it/s]Generating embeddings for dataset:  31%|███       | 269/875 [00:01<00:04, 147.65it/s]Generating embeddings for dataset:  32%|███▏      | 284/875 [00:01<00:04, 146.73it/s]Generating embeddings for dataset:  34%|███▍      | 299/875 [00:01<00:03, 146.25it/s]Generating embeddings for dataset:  36%|███▌      | 314/875 [00:02<00:03, 145.81it/s]Generating embeddings for dataset:  38%|███▊      | 329/875 [00:02<00:03, 145.44it/s]Generating embeddings for dataset:  39%|███▉      | 344/875 [00:02<00:03, 145.09it/s]Generating embeddings for dataset:  41%|████      | 359/875 [00:02<00:03, 144.39it/s]Generating embeddings for dataset:  43%|████▎     | 374/875 [00:02<00:03, 143.76it/s]Generating embeddings for dataset:  44%|████▍     | 389/875 [00:02<00:03, 143.46it/s]Generating embeddings for dataset:  46%|████▌     | 404/875 [00:02<00:03, 143.05it/s]Generating embeddings for dataset:  48%|████▊     | 419/875 [00:02<00:03, 142.63it/s]Generating embeddings for dataset:  50%|████▉     | 434/875 [00:02<00:03, 141.79it/s]Generating embeddings for dataset:  51%|█████▏    | 449/875 [00:03<00:03, 141.60it/s]Generating embeddings for dataset:  53%|█████▎    | 464/875 [00:03<00:02, 140.50it/s]Generating embeddings for dataset:  55%|█████▍    | 479/875 [00:03<00:02, 139.58it/s]Generating embeddings for dataset:  56%|█████▋    | 493/875 [00:03<00:02, 139.65it/s]Generating embeddings for dataset:  58%|█████▊    | 507/875 [00:03<00:02, 139.13it/s]Generating embeddings for dataset:  60%|█████▉    | 521/875 [00:03<00:03, 112.14it/s]Generating embeddings for dataset:  61%|██████    | 534/875 [00:03<00:02, 115.51it/s]Generating embeddings for dataset:  63%|██████▎   | 548/875 [00:03<00:02, 121.58it/s]Generating embeddings for dataset:  64%|██████▍   | 562/875 [00:03<00:02, 126.08it/s]Generating embeddings for dataset:  66%|██████▌   | 576/875 [00:04<00:02, 129.71it/s]Generating embeddings for dataset:  67%|██████▋   | 590/875 [00:04<00:02, 131.89it/s]Generating embeddings for dataset:  69%|██████▉   | 604/875 [00:04<00:02, 133.70it/s]Generating embeddings for dataset:  71%|███████   | 618/875 [00:04<00:01, 134.30it/s]Generating embeddings for dataset:  72%|███████▏  | 632/875 [00:04<00:01, 135.21it/s]Generating embeddings for dataset:  74%|███████▍  | 646/875 [00:04<00:01, 135.61it/s]Generating embeddings for dataset:  75%|███████▌  | 660/875 [00:04<00:01, 135.75it/s]Generating embeddings for dataset:  77%|███████▋  | 674/875 [00:04<00:01, 135.60it/s]Generating embeddings for dataset:  79%|███████▊  | 688/875 [00:04<00:01, 135.48it/s]Generating embeddings for dataset:  80%|████████  | 702/875 [00:04<00:01, 135.34it/s]Generating embeddings for dataset:  82%|████████▏ | 716/875 [00:05<00:01, 135.13it/s]Generating embeddings for dataset:  83%|████████▎ | 730/875 [00:05<00:01, 108.77it/s]Generating embeddings for dataset:  85%|████████▍ | 742/875 [00:05<00:01, 110.46it/s]Generating embeddings for dataset:  86%|████████▋ | 755/875 [00:05<00:01, 115.09it/s]Generating embeddings for dataset:  88%|████████▊ | 768/875 [00:05<00:00, 118.61it/s]Generating embeddings for dataset:  89%|████████▉ | 781/875 [00:05<00:00, 120.85it/s]Generating embeddings for dataset:  91%|█████████ | 794/875 [00:05<00:00, 122.84it/s]Generating embeddings for dataset:  92%|█████████▏| 807/875 [00:05<00:00, 124.04it/s]Generating embeddings for dataset:  94%|█████████▎| 820/875 [00:05<00:00, 124.87it/s]Generating embeddings for dataset:  95%|█████████▌| 833/875 [00:06<00:00, 125.55it/s]Generating embeddings for dataset:  97%|█████████▋| 846/875 [00:06<00:00, 125.75it/s]Generating embeddings for dataset:  98%|█████████▊| 859/875 [00:06<00:00, 126.09it/s]Generating embeddings for dataset: 100%|█████████▉| 872/875 [00:06<00:00, 125.65it/s]Generating embeddings for dataset: 100%|██████████| 875/875 [00:06<00:00, 136.64it/s]
Function 'generate' executed in 121.6576s
Generating embeddings for dataset:   0%|          | 0/160 [00:00<?, ?it/s]Generating embeddings for dataset:   8%|▊         | 13/160 [00:00<00:01, 122.17it/s]Generating embeddings for dataset:  16%|█▋        | 26/160 [00:00<00:01, 122.06it/s]Generating embeddings for dataset:  24%|██▍       | 39/160 [00:00<00:00, 122.87it/s]Generating embeddings for dataset:  32%|███▎      | 52/160 [00:00<00:00, 122.27it/s]Generating embeddings for dataset:  41%|████▏     | 66/160 [00:00<00:00, 125.26it/s]Generating embeddings for dataset:  49%|████▉     | 79/160 [00:00<00:00, 125.20it/s]Generating embeddings for dataset:  57%|█████▊    | 92/160 [00:00<00:00, 124.53it/s]Generating embeddings for dataset:  66%|██████▌   | 105/160 [00:00<00:00, 123.95it/s]Generating embeddings for dataset:  74%|███████▍  | 118/160 [00:00<00:00, 123.66it/s]Generating embeddings for dataset:  82%|████████▏ | 131/160 [00:01<00:00, 123.55it/s]Generating embeddings for dataset:  90%|█████████ | 144/160 [00:01<00:00, 123.20it/s]Generating embeddings for dataset:  98%|█████████▊| 157/160 [00:01<00:00, 123.00it/s]Generating embeddings for dataset: 100%|██████████| 160/160 [00:01<00:00, 123.80it/s]
Function 'generate' executed in 23.3857s
