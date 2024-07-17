# SIDE

The code source of "SIDE: Sequence-Interaction-Aware Dual Encoder for Predicting circRNA Back-Splicing Events"
accepted by IEEE BIBM 2023.

## Dependencies

Compatible with PyTorch 1.0 and Python 3.x.

Dependencies can be installed using the requirements.txt file.

## Required Packages

numpy

pandas

tqdm

scikit-learn

torch

subword-nmt

## Run

You can directly run 'python train.py' to run the experiments.

## Parameter Setting

You can set the parameters in “config.py”.

You can choose how to encode the input sequence in “stream.py”.

## Example

$ cd SIDE/

$ python train.py

--- Data Preparation ---

crossfold0-----------------------------------------------------------

Initial Testing AUROC: 0.5398638200906057 , AUPRC: 0.5304521200013526 , F1: 0.6285714285714286 , Test loss: 0.7075778245925903Test acc0.4921875Test Precion0.49698795180722893Test recall0.8549222797927462

--- Go for Training ---

Training at Epoch 1 iteration 0 with loss 0.7244067

Training at Epoch 1 , AUROC: 0.8900628346222487 , AUPRC: 0.898391384149437 , F1: 0.8032733224222586Test acc0.8167682926829268Test Precion0.8671378091872791Test recall0.7481707317073171

Validation at Epoch 1 , AUROC: 0.8237961225766105 , AUPRC: 0.8179783344037446 , F1: 0.6986301369863015Test acc0.725Test Precion0.75Test recall0.6538461538461539

Testing AUROC: 0.864990234375 , AUPRC: 0.8906058695351841 , F1: 0.7734806629834253 , Test loss: 0.48601171374320984Test acc0.7864583333333334Test Precion0.8235294117647058Test recall0.7291666666666666

Training at Epoch 2 iteration 0 with loss 0.43935466

Training at Epoch 2 , AUROC: 0.8471538518738846 , AUPRC: 0.8662154522656732 , F1: 0.7674487674487674Test acc0.761280487804878Test Precion0.7481181239143022Test recall0.7878048780487805

Validation at Epoch 2 , AUROC: 0.8086303939962477 , AUPRC: 0.8300666025365329 , F1: 0.7017543859649121Test acc0.68125Test Precion0.6741573033707865Test recall0.7317073170731707

Testing AUROC: 0.8297479857852046 , AUPRC: 0.8536685875344361 , F1: 0.7676767676767676 , Test loss: 0.5100747346878052Test acc0.7604166666666666Test Precion0.7487684729064039Test recall0.7875647668393783

Training at Epoch 3 iteration 0 with loss 0.5501201

Training at Epoch 3 , AUROC: 0.9420783759666864 , AUPRC: 0.9325374426889235 , F1: 0.869455803711859Test acc0.8734756097560976Test Precion0.8979857050032488Test recall0.8426829268292683

Validation at Epoch 3 , AUROC: 0.8653124999999999 , AUPRC: 0.8319293334085806 , F1: 0.7922077922077922Test acc0.8Test Precion0.8243243243243243Test recall0.7625

Testing AUROC: 0.8939315845156389 , AUPRC: 0.9090855552192154 , F1: 0.8181818181818181 , Test loss: 0.40337294340133667Test acc0.8229166666666666Test Precion0.8453038674033149Test recall0.7927461139896373

Training at Epoch 4 iteration 0 with loss 0.267914

Training at Epoch 4 , AUROC: 0.947792236763831 , AUPRC: 0.9529540048178903 , F1: 0.7169149868536373Test acc0.6060975609756097Test Precion0.5595075239398085Test recall0.9975609756097561

Validation at Epoch 4 , AUROC: 0.8526332239412409 , AUPRC: 0.8340393332438649 , F1: 0.6968325791855204Test acc0.58125Test Precion0.5422535211267606Test recall0.9746835443037974

Testing AUROC: 0.8927922306920218 , AUPRC: 0.9040342496189339 , F1: 0.6994535519125683 , Test loss: 1.025489091873169Test acc0.5703125Test Precion0.5393258426966292Test recall0.9948186528497409

Training at Epoch 5 iteration 0 with loss 0.23720904

Training at Epoch 5 , AUROC: 0.9505465312858906 , AUPRC: 0.9493055260569381 , F1: 0.8758252121974223Test acc0.8795731707317073Test Precion0.9033722438391699Test recall0.849908480780964

Validation at Epoch 5 , AUROC: 0.875 , AUPRC: 0.8782528034770642 , F1: 0.8098159509202455Test acc0.80625Test Precion0.7951807228915663Test recall0.825

Testing AUROC: 0.8731553819444444 , AUPRC: 0.8600703621137912 , F1: 0.7798408488063661 , Test loss: 0.46166926622390747Test acc0.7838541666666666Test Precion0.7945945945945946Test recall0.765625

Training at Epoch 6 iteration 0 with loss 0.25128946

Training at Epoch 6 , AUROC: 0.9341983693479958 , AUPRC: 0.8465991131293185 , F1: 0.8886123210952084Test acc0.8908536585365854Test Precion0.9078194532739987Test recall0.870201096892139

Validation at Epoch 6 , AUROC: 0.8649155722326454 , AUPRC: 0.8607078398999877 , F1: 0.7534246575342467Test acc0.775Test Precion0.8088235294117647Test recall0.7051282051282052

Testing AUROC: 0.8834061253831755 , AUPRC: 0.8213344338787825 , F1: 0.8297872340425533 , Test loss: 1.4581890106201172Test acc0.8333333333333334Test Precion0.8524590163934426Test recall0.8082901554404145

Training at Epoch 7 iteration 0 with loss 0.39096573

Training at Epoch 7 , AUROC: 0.9767089122472524 , AUPRC: 0.9444131563191062 , F1: 0.9300590612371774Test acc0.9314024390243902Test Precion0.9468354430379747Test recall0.9138668295662797

Validation at Epoch 7 , AUROC: 0.8788305190744216 , AUPRC: 0.8421608857958963 , F1: 0.782608695652174Test acc0.78125Test Precion0.7974683544303798Test recall0.7682926829268293

Testing AUROC: 0.9097357855902778 , AUPRC: 0.8858873599066359 , F1: 0.8337874659400546 , Test loss: 0.6503826379776001Test acc0.8411458333333334Test Precion0.8742857142857143Test recall0.796875

Training at Epoch 8 iteration 0 with loss 0.2033371

Training at Epoch 8 , AUROC: 0.9666837820322867 , AUPRC: 0.9426164426983753 , F1: 0.9153046062407133Test acc0.913109756097561Test Precion0.8912037037037037Test recall0.9407452657299938

Validation at Epoch 8 , AUROC: 0.8723238005938427 , AUPRC: 0.8550865810379318 , F1: 0.8048780487804879Test acc0.8Test Precion0.7764705882352941Test recall0.8354430379746836

Testing AUROC: 0.9075224479830724 , AUPRC: 0.9077397121962265 , F1: 0.8329048843187661 , Test loss: 0.3972393870353699Test acc0.8307291666666666Test Precion0.826530612244898Test recall0.8393782383419689

Training at Epoch 9 iteration 0 with loss 0.19722798

Training at Epoch 9 , AUROC: 0.9823607534059912 , AUPRC: 0.9770358674338371 , F1: 0.9247437774524158Test acc0.9216463414634146Test Precion0.8890765765765766Test recall0.963392312385601

Validation at Epoch 9 , AUROC: 0.8951650758879675 , AUPRC: 0.8837666265871618 , F1: 0.8415300546448088Test acc0.81875Test Precion0.77Test recall0.927710843373494

Testing AUROC: 0.9100206163194444 , AUPRC: 0.8903848549992969 , F1: 0.8382352941176471 , Test loss: 0.41079777479171753Test acc0.828125Test Precion0.7916666666666666Test recall0.890625

Training at Epoch 10 iteration 0 with loss 0.11760622

Training at Epoch 10 , AUROC: 0.9884856312992732 , AUPRC: 0.9871410611964677 , F1: 0.9388468258590564Test acc0.9359756097560976Test Precion0.8975501113585747Test recall0.9841269841269841

Validation at Epoch 10 , AUROC: 0.8807066916823014 , AUPRC: 0.8490479478673443 , F1: 0.8095238095238095Test acc0.8Test Precion0.7555555555555555Test recall0.8717948717948718

Testing AUROC: 0.9211268751865014 , AUPRC: 0.904912352461964 , F1: 0.8489208633093526 , Test loss: 0.6530150175094604Test acc0.8359375Test Precion0.7901785714285714Test recall0.917098445595855

--- Go for Testing ---

Testing AUROC: 0.8928629557291666 , AUPRC: 0.8391614376013927 , F1: 0.8324324324324325 , Test loss: 1.2289551496505737Test acc0.8385416666666666Test Precion0.8651685393258427Test recall0.8020833333333334

crossfold1-----------------------------------------------------------

Initial Testing AUROC: 0.4378255208333333 , AUPRC: 0.46648384742776783 , F1: 0.526077097505669 , Test loss: 0.7027361392974854Test acc0.4557291666666667Test Precion0.46586345381526106Test recall0.6041666666666666

--- Go for Training ---

xxxx

xxxx

avg_auc 0.8981748703588828

avg_auprc 0.881937401738365

avg_acc 0.8268229166666667

avg_pre 0.8336204698799377

avg_recall 0.8259367737749687

avg_f1 0.8256604572030476

11403.612461090088

