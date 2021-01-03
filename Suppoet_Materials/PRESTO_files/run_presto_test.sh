#!/bash/bin

export PRESTO=/ASC/presto-master
export TEMPO=/ASC/tempo
export PGPLOT_DIR=/usr/lib/pgplot5
export PATH=$PATH:/ASC/presto-master/bin

cd /ASC/TestData1/
(time python2 ./pipeline.py GBT_Lband_PSR.fil) > log.pulsar_search 2>&1
cd /ASC/TestData2/
(time python2 ./pipeline.py Dec+1554_arcdrift+23.4-M12_0194.fil) > log.pulsar_search 2>&1

exit 0
