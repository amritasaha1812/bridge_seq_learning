# bridge_seq_learning

This repository has the tensorflow code of the paper "A Correlational Encoder Decoder Architecture for Pivot Based Sequence Generation"  accepted at COLING 2016.

The idea is to use neural encoder decoder architectures to encode multiple languages into a common linguistic representation and then decode sentences in multiple target languages from this repre- sentation.
 Specifically, we consider the case of three languages or modalities X, Z and Y wherein we are interested in generating sequences in Y starting from information available in X. However, there is no parallel training data available between X and Y but, training data is available between X & Z and Z & Y (as is often the case in many real world applications). Z thus acts as a pivot/bridge. An obvious solution, which is perhaps less elegant but works very well in practice is to train a two stage model which first converts from X to Z and then from Z to Y . Instead we explore an interlingua inspired solu- tion which jointly learns to do the following (i) encode X and Z to a common representation and (ii) decode Y from this common representation. 

There are two settings in which this archhitecture was tried out, (i) bridge transliteration and (ii) bridge captioning

