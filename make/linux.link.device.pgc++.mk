LINK := $(ACC)
link_flags__ += $(acc_opt_flags__)
link_flags__ += -acc -Mcudalib=cufft,cublas
