
use insofe_customerdata;
CREATE TABLE predictions (
  customer_no int(11) NOT NULL,
  prediction varchar(250) NOT NULL,
  age int(11) DEFAULT NULL,
  job varchar(250) DEFAULT NULL,
  marital varchar(250) DEFAULT NULL,
  eduation varchar(250) DEFAULT NULL,
  credit_default varchar(250) DEFAULT NULL,
  housing varchar(30) DEFAULT NULL,
  loan varchar(250) DEFAULT NULL,
  contact varchar(250) DEFAULT NULL,
  contacted_month varchar(250) DEFAULT NULL,
  day_of_week varchar(250) DEFAULT NULL,
  duration int(11) DEFAULT NULL,
  compaign varchar(250) DEFAULT NULL,
  pdays int(11) DEFAULT NULL,
  previous int(11) DEFAULT NULL,
  poutcome varchar(250) DEFAULT NULL,
  emp_var_rate double DEFAULT NULL,
  cons_price_idx double DEFAULT NULL,
  cons_conf_idx double DEFAULT NULL,
  euribor3m double DEFAULT NULL,
  nr_employees double DEFAULT NULL,
  PRIMARY KEY (customer_no)
);