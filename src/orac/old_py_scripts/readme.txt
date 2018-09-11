This is just the code that we will use to run ORAC.

Some notes on what I understand of the code so far:
    Parser defines all the processing parameters. They
    can be accessed in the scripts via args.var and assigned
    new values as needed.

    Most of the processing parameters will not need to be changed
    however, args.target is useful to set as it defines the file that
    is going to be processed.  If I do not use the batch system,
    but instead go for multiprocessing this could be very useful. Also,
    limit defines the roi of the rile that is going to be processed.  If
    it is set to all zero, then the entire file is processed.

    The local defaults file sets most of the processing parameters.