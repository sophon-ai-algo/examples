# -*- coding:utf-8 -*-
import redis
import time
import multiprocessing
from utils import logger

log = logger.get_logger(__file__)



class RedisClientInstance(object):
    # 线程锁
    _instance_lock = multiprocessing.Lock()

    def __init__(self, *args,**kwargs):
        pass

    @classmethod
    def get_storage_instance(cls):
        if not hasattr(RedisClientInstance,'_instance'):
            with RedisClientInstance._instance_lock:
                RedisClientInstance._instance = RedisClient(host="127.0.0.1",
                                                            port=6379,
                                                            db=1,
                                                            password=None)
            log.info('================= redis client instance  =====================')
        return RedisClientInstance._instance



class RedisClient:
    def __init__(self, host, port, password, db=0, decode_responses=False):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.decode_responses = decode_responses
        self.socket_timeout = 5
        self.socket_connect_timeout = 2
        self.max_connections = 100
        self.redis_client = None
        self.connect()


    def connect(self):
        try:
            log.info(' ================= redis  connect =====================')
            connect_params = {
                'host': self.host,
                'port': self.port,
                'password': self.password,
                'db': self.db,
                'decode_responses': self.decode_responses,
                'socket_timeout': self.socket_timeout,
                'socket_connect_timeout': self.socket_connect_timeout,
            }

            if self.password:
                connect_params['password'] = self.password

            pool = redis.ConnectionPool(max_connections=self.max_connections, **connect_params)
            self.redis_client = redis.Redis(connection_pool=pool)
            return True
        except:
            return False

    # 重新连接
    def re_connect(self):
        _number = 0
        _status = True
        # 尝试重连
        while _status:
            try:
                self.redis_client.ping()
                _status = False
            except:
                log.info('断开重连 {}'.format(time.time()))
                if self.connect():
                    _status = False
                    break
                # 连接不成功，休息三秒
                time.sleep(3)


    # 获取Redis服务器的时间 UNIX时间戳 + 这一秒已经逝去的微秒数
    def get_time(self):
        try:
            self.re_connect()
            return self.redis_client.time()
        except:
            log.info('{}  执行失败'.format('Redis get time'))


    # 获取所有对象名称
    def get_all_keys(self, pattern='*'):
        try:
            self.re_connect()
            return self.redis_client.keys(pattern=pattern)

        except:
            log.info('{}  执行失败'.format('Redis get all keys'))

    # 随机获取一个Key
    def get_random_key(self):
        try:
            self.re_connect()
            return self.redis_client.randomkey()

        except:
            log.info('{}  执行失败'.format('Redis get randomkey'))

    # 设置Key在固定时间之后过期
    def set_key_expire_after(self, name, unit, time):
        try:
            self.re_connect()
            if unit == "second":
                return self.redis_client.expire(name, time)
            elif unit == "millisecond":
                return self.redis_client.pexpire(name, time)

        except:
            log.info('{}  执行失败'.format('Redis set_key_expire_after'))


    # 设置Key在到达固定时间之时过期
    def set_key_expire_at(self, name, unit, time):
        try:
            self.re_connect()
            if unit == "second":
                return self.redis_client.expireat(name, time)
            elif unit == "millisecond":
                return self.redis_client.pexpireat(name, time)

        except:
            log.info('{}  执行失败'.format('Redis set_key_expire_at'))


    # 重新命名对象
    def rename_key(self, old_name, new_name):
        try:
            self.re_connect()
            return self.redis_client.rename(old_name, new_name)

        except:
            log.info('{}  执行失败'.format('Redis rename_key'))


    # 新Key不存在的情况下重命名
    def rename_key_if_not_exists(self, old_name, new_name):
        try:
            self.re_connect()
            return self.redis_client.renamenx(old_name, new_name)

        except:
            log.info('{}  执行失败'.format('Redis rename_key'))

    # 获取对象的类型
    def get_type_of_object(self, name):
        try:
            self.re_connect()
            return self.redis_client.type(name)

        except:
            log.info('{}  执行失败'.format('Redis get_type_of_object'))

    # 设置对象为永久保存
    def persist(self, name):
        try:
            self.re_connect()
            return self.redis_client.persist(name)

        except:
            log.info('{}  执行失败'.format('Redis persist'))

    # 将对象移动到指定数据库
    def move_key_to_database(self, name, db):
        try:
            self.re_connect()
            return self.redis_client.move(name, db)

        except:
            log.info('{}  执行失败'.format('Redis move_key_to_database'))

    # 获取对象的剩余存活时间
    def get_time_to_live(self, name, unit):
        try:
            self.re_connect()

            if unit == "second":
                return self.redis_client.ttl(name)
            elif unit == "millisecond":
                return self.redis_client.pttl(name)

        except:
            log.info('{}  执行失败'.format('Redis get_time_to_live'))

    # 判断是否有某个名字的对象
    def exists(self, name):
        try:
            self.re_connect()
            return self.redis_client.exists(name)

        except:
            log.info('{}  执行失败'.format('Redis exists'))


    # 删除对象
    def delete_by_name(self, name):
        try:
            self.re_connect()
            return self.redis_client.delete(name)

        except:
            log.info('{}  执行失败'.format('Redis delete_by_name'))


    def get_by_name(self, name):
        try:
            self.re_connect()
            return self.redis_client.get(name)

        except:
            log.info('{}  执行失败'.format('Redis get_by_name'))

    # 删除指定数据库中的内容
    def flush_database(self):
        try:
            self.re_connect()
            return self.redis_client.flushdb()

        except:
            log.info('{}  执行失败'.format('Redis flush_database'))

    # 删除整个Redis中的内容
    def flush_all(self):
        try:
            self.re_connect()
            return self.redis_client.flushall()

        except:
            log.info('{}  执行失败'.format('Redis flush_all'))

    def get_used_memory(self):
        try:
            self.re_connect()
            return self.redis_client.info()['used_memory']

        except:
            log.info('{}  执行失败'.format('Redis get_used_memory'))


    def get_maximum_memory(self):
        try:
            self.re_connect()
            return self.redis_client.info()['maxmemory']

        except:
            log.info('{}  执行失败'.format('Redis get_maximum_memory'))


    def get_db_save(self):
        try:
            self.re_connect()
            return self.redis_client.config_get('save')['save']

        except:
            log.info('{}  执行失败'.format('Redis get_db_save'))


    def get_db_databases(self):
        try:
            self.re_connect()
            return self.redis_client.config_get('databases')['databases']

        except:
            log.info('{}  执行失败'.format('Redis get_db_databases'))

    def get_db_size(self):
        try:
            self.re_connect()
            return self.redis_client.dbsize()

        except:
            log.info('{}  执行失败'.format('Redis get_db_size'))


    def set_db_config(self, key, value):
        try:
            bool = self.redis_client.config_set(key, value)
            return bool
        except:
            return False

    def set_db_config_file(self, key, value):
        try:
            bool = self.redis_client.config_set(key, value)
            self.redis_client.config_rewrite()
            return bool
        except:
            return False
    '''
    String
    '''

    # 单个添加字符串
    def single_set_string(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.set(name, value)

        except:
            log.info('{}  执行失败'.format('Redis single_set_string'))


    # 批量添加字符串
    def multi_set_string(self, *args, **kwargs):
        try:
            self.re_connect()
            return self.redis_client.mset(*args, **kwargs)

        except:
            log.info('{}  执行失败'.format('Redis multi_set_string'))


    # 获取单个字符串
    def single_get_string(self, name):
        try:
            self.re_connect()
            return self.redis_client.get(name)

        except:
            log.info('{}  执行失败'.format('Redis single_get_string'))


    # 批量获取字符串
    def multi_get_string(self, keys):
        try:
            self.re_connect()
            return self.redis_client.mget(keys)

        except:
            log.info('{}  执行失败'.format('Redis multi_get_string'))


    # 设置字符串并设置Key过期时间
    def single_set_string_with_expire_time(self, key, unit, time, value):
        try:
            self.re_connect()
            if unit == "second":
                return self.redis_client.setex(key, time, value)
            elif unit == "millisecond":
                return self.redis_client.psetex(key, time, value)
        except:
            log.info('{}  执行失败'.format('Redis single_set_string_with_expire_time'))


    # 将给定 name 的值设为 value ，并返回 name 的旧值(old value)
    def get_and_set_string(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.getset(name, value)

        except:
            log.info('{}  执行失败'.format('Redis get_and_set_string'))


    # 获取指定区间的字符串
    def get_string_by_range(self, name, start, end):
        try:
            self.re_connect()
            return self.redis_client.getrange(name, start, end)

        except:
            log.info('{}  执行失败'.format('Redis get_string_by_range'))

    # 获取字符串的长度
    def get_string_length(self, name):
        try:
            self.re_connect()
            return self.redis_client.strlen(name)

        except:
            log.info('{}  执行失败'.format('Redis get_string_length'))


    # 将String中指定偏移量的字符串更换为指定字符串
    def set_string_by_range(self, name, offset, value):
        try:
            self.re_connect()
            return self.redis_client.setrange(name, offset, value)

        except:
            log.info('{}  执行失败'.format('Redis set_string_by_range'))

    # 添加字符串到尾部
    def append_to_string_tail(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.append(name, value)

        except:
            log.info('{}  执行失败'.format('Redis append_to_string_tail'))

    # Key中存储的数字值按给定值增加
    def increase_by(self, name, type="int", increment=1):
        try:
            self.re_connect()
            if type == "int":
                return self.redis_client.incrby(name, increment)
            elif type == "float":
                return self.redis_client.incrbyfloat(name, increment)

        except:
            log.info('{}  执行失败'.format('Redis increase_by'))


    # Key中存储的数字值按给定值减少
    def decrease_by(self, name, decrement=1):
        try:
            self.re_connect()
            return self.redis_client.decrby(name, decrement)

        except:
            log.info('{}  执行失败'.format('Redis decrease_by'))


    '''
    Set
    '''

    # 添加元素到Set
    def add_set(self, name, *values):
        try:
            self.re_connect()
            return self.redis_client.sadd(name, *values)

        except:
            log.info('{}  执行失败'.format('Redis add_set'))

    # 取得Set中的元素个数
    def get_length_of_set(self, name):
        try:
            self.re_connect()
            return self.redis_client.scard(name)

        except:
            log.info('{}  执行失败'.format('Redis get_length_of_set'))


    # 获取Set中的所有元素
    def get_all_members_of_set(self, name):
        try:
            self.re_connect()
            return self.redis_client.smembers(name)

        except:
            log.info('{}  执行失败'.format('Redis get_all_members_of_set'))


    # 随机获取Set中指定个数的元素
    def get_random_members_of_set(self, name, number):
        try:
            self.re_connect()
            return self.redis_client.srandmember(name, number)

        except:
            log.info('{}  执行失败'.format('Redis get_random_members_of_set'))


    # 第一个Set里有而后面的Set/s里没有的元素
    def get_diff_members_from_sets(self, keys, *set_names):
        try:
            self.re_connect()
            return self.redis_client.sdiff(keys, *set_names)

        except:
            log.info('{}  执行失败'.format('Redis get_diff_members_from_sets'))


    # 将第一个Set里有而后面的Set/s里没有的元素存到一个新的Set中
    def add_diff_members_from_sets_to_new_set(self, dest, keys, *set_names):
        try:
            self.re_connect()
            return self.redis_client.sdiffstore(dest, keys, *set_names)

        except:
            log.info('{}  执行失败'.format('Redis add_diff_members_from_sets_to_new_set'))

    # 随机获取并删除Set中的一个元素
    def pop_member_of_set(self, name):
        try:
            self.re_connect()
            return self.redis_client.spop(name)

        except:
            log.info('{}  执行失败'.format('Redis pop_member_of_set'))

    # 判断Set中是否有某个元素
    def exist_member_in_set(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.sismember(name, value)

        except:
            log.info('{}  执行失败'.format('Redis exist_member_in_set'))

    '''
    List
    '''

    # 获取List中元素个数
    def len_of_list(self, name):
        try:
            self.re_connect()
            return self.redis_client.llen(name)

        except:
            log.info('{}  执行失败'.format('Redis len_of_list'))

    # 插入元素到List头部
    def insert_to_list_head(self, name, *values):
        try:
            self.re_connect()
            return self.redis_client.lpush(name, *values)

        except:
            log.info('{}  执行失败'.format('Redis insert_to_list_head'))

    # 插入元素到List尾部
    def insert_to_list_tail(self, name, *values):
        try:
            self.re_connect()
            return self.redis_client.rpush(name, *values)

        except:
            log.info('{}  执行失败'.format('Redis insert_to_list_tail'))


    # 从List头部获取一个元素并删除
    def get_and_delete_from_list_head(self, name):
        try:
            self.re_connect()
            return self.redis_client.lpop(name)

        except:
            log.info('{}  执行失败'.format('Redis get_and_delete_from_list_head'))

    # 从List尾部获取一个元素并删除
    def get_and_delete_from_list_tail(self, name):
        try:
            self.re_connect()
            return self.redis_client.rpop(name)

        except:
            log.info('{}  执行失败'.format('Redis get_and_delete_from_list_tail'))


    # 移出并获取列表的第一个元素， 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止
    def block_and_pop_first_item_in_list(self, keys, timeout=1):
        try:
            self.re_connect()
            return self.redis_client.blpop(keys, timeout=timeout)

        except:
            log.info('{}  执行失败'.format('Redis block_and_pop_first_item_in_list'))

    # 移出并获取列表的最后一个元素， 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止
    def block_and_pop_last_item_in_list(self, keys, timeout=1):
        try:
            self.re_connect()
            return self.redis_client.brpop(keys, timeout=timeout)

        except:
            log.info('{}  执行失败'.format('Redis block_and_pop_last_item_in_list'))

    # 从列表中弹出一个值，将弹出的元素插入到另外一个列表中并返回它， 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止
    def block_pop_item_from_list_and_push_to_list(self, src, dst, timeout=1):
        try:
            self.re_connect()
            return self.redis_client.brpoplpush(src, dst, timeout=timeout)

        except:
            log.info('{}  执行失败'.format('Redis block_pop_item_from_list_and_push_to_list'))

    # 从列表中弹出一个值，将弹出的元素插入到另外一个列表中并返回它
    def pop_item_from_list_and_push_to_list(self, src, dst):
        try:
            self.re_connect()
            return self.redis_client.rpoplpush(src, dst)

        except:
            log.info('{}  执行失败'.format('Redis pop_item_from_list_and_push_to_list'))

    # 获取List中的元素
    def get_item_in_list_by_index(self, name, index):
        try:
            self.re_connect()
            return self.redis_client.lindex(name, index)

        except:
            log.info('{}  执行失败'.format('Redis get_item_in_list_by_index'))

    # 将一个值插入到已存在的列表头部
    def insert_item_to_the_head_of_existent_list(self, name, item):
        try:
            self.re_connect()
            return self.redis_client.lpushx(name, item)

        except:
            log.info('{}  执行失败'.format('Redis insert_item_to_the_head_of_existent_list'))

    # 将一个值插入到已存在的列表尾部
    def insert_item_to_the_tail_of_existent_list(self, name, item):
        try:
            self.re_connect()
            return self.redis_client.rpushx(name, item)

        except:
            log.info('{}  执行失败'.format('Redis insert_item_to_the_tail_of_existent_list'))

    # 获取List中指定区间的元素
    def get_items_in_list_by_range(self, name, start, stop):
        try:
            self.re_connect()
            return self.redis_client.lrange(name, start, stop)

        except:
            log.info('{}  执行失败'.format('Redis get_items_in_list_by_range'))


    # 删除List中的元素
    def delete_item_in_list(self, name, count, value):
        """
        Remove the first ``count`` occurrences of elements equal to ``value``
        from the list stored at ``name``.

        The count argument influences the operation in the following ways:
           count > 0: Remove elements equal to value moving from head to tail.
           count < 0: Remove elements equal to value moving from tail to head.
           count = 0: Remove all elements equal to value.
        """
        try:
            self.re_connect()
            return self.redis_client.lrem(name, count, value)

        except:
            log.info('{}  执行失败'.format('Redis delete_item_in_list'))

    # 根据索引设置List中的元素
    def set_item_in_list_by_index(self, name, index):
        try:
            self.re_connect()
            return self.redis_client.lset(name, index)

        except:
            log.info('{}  执行失败'.format('Redis set_item_in_list_by_index'))


    #  保留List中指定区间元素
    def trim_items_in_list_by_range(self, name, start, end):
        try:
            self.re_connect()
            return self.redis_client.ltrim(name, start, end)

        except:
            log.info('{}  执行失败'.format('Redis trim_items_in_list_by_range'))

    '''
    Hash
    '''

    # 删除Hash中的某字段
    def delete_field_in_hash(self, name, *filed_name):
        try:
            self.re_connect()
            return self.redis_client.hdel(name, *filed_name)

        except:
            log.info('{}  执行失败'.format('Redis delete_field_in_hash'))

    # 判断Hash中是否有某字段
    def exists_field_in_hash(self, name, field_name):
        try:
            self.re_connect()
            return self.redis_client.hexists(name, field_name)

        except:
            log.info('{}  执行失败'.format('Redis exists_field_in_hash'))

    # 获取Hash中的某字段值
    def get_field_in_hash(self, name, field_name):
        try:
            self.re_connect()
            return self.redis_client.hget(name, field_name)

        except:
            log.info('{}  执行失败'.format('Redis get_field_in_hash'))

    # 获取Hash中所有字段名称和值
    def get_all_fields_in_hash(self, name):
        try:
            self.re_connect()
            return self.redis_client.hgetall(name)

        except:
            log.info('{}  执行失败'.format('Redis get_all_fields_in_hash'))

    # 增加Hash中某字段值
    def increase_by_field_in_hash(self, name, field_name, type="int", increment=1):
        try:
            self.re_connect()
            if type == "int":
                return self.redis_client.hincrby(name, field_name, increment)
            elif type == "float":
                return self.redis_client.hincrbyfloat(name, field_name, increment)
        except:
            log.info('{}  执行失败'.format('Redis increase_by_field_in_hash'))


    # 获取Hash中所有值
    def get_all_values_in_hash(self, name):
        try:
            self.re_connect()
            return self.redis_client.hvals(name)

        except:
            log.info('{}  执行失败'.format('Redis get_all_values_in_hash'))


    # 获取Hash中字段的个数
    def get_length_of_fields_in_hash(self, name):
        try:
            self.re_connect()
            return self.redis_client.hlen(name)

        except:
            log.info('{}  执行失败'.format('Redis get_length_of_fields_in_hash'))

    # 批量获取Hash中的字段
    def multi_get_fields_in_hash(self, name, keys, *args):
        try:
            self.re_connect()
            return self.redis_client.hmget(name, keys, *args)

        except:
            log.info('{}  执行失败'.format('Redis multi_get_fields_in_hash'))


    # 批量设置Hash中的字段
    def multi_set_fields_in_hash(self, name, *field_name, **field_values):
        try:
            self.re_connect()
            return self.redis_client.hmset(name, *field_name, **field_values)

        except:
            log.info('{}  执行失败'.format('Redis multi_set_fields_in_hash'))

    # 单个设置Hash中的字段
    def single_set_filed_in_hash(self, name, field, value):
        try:
            self.re_connect()
            return self.redis_client.hset(name, field, value)

        except:
            log.info('{}  执行失败'.format('Redis single_set_filed_in_hash'))


    # Hash中不存在某字段时设置该字段
    def set_field_if_not_exists_in_hash(self, name, field, value):
        try:
            self.re_connect()
            return self.redis_client.hsetnx(name, field, value)

        except:
            log.info('{}  执行失败'.format('Redis set_field_if_not_exists_in_hash'))

    # 获取Hash中所有字段的名称
    def get_all_field_name_in_hash(self, name):
        try:
            self.re_connect()
            return self.redis_client.hkeys(name)

        except:
            log.info('{}  执行失败'.format('Redis get_all_field_name_in_hash'))

    '''
    Sorted Set
    '''

    # 向有序集合添加一个或多个成员，或者更新已存在成员的分数
    def z_add_members(self, name, *args, **kwargs):
        try:
            self.re_connect()
            return self.redis_client.zadd(name, *args, **kwargs)

        except:
            log.info('{}  执行失败'.format('Redis z_add_members'))

    # 获取有序集合的成员数
    def z_get_count_of_members(self, name):
        try:
            self.re_connect()
            return self.redis_client.zcard(name)

        except:
            log.info('{}  执行失败'.format('Redis z_get_count_of_members'))

    # 计算在有序集合中指定区间分数的成员数
    def z_get_count_of_members_between_scores(self, name, min, max):
        try:
            self.re_connect()
            return self.redis_client.zcount(name, min, max)

        except:
            log.info('{}  执行失败'.format('Redis z_get_count_of_members_between_scores'))


    # 有序集合中对指定成员的分数加上增量 increment
    def z_increase_member(self, name, increment=1):
        try:
            self.re_connect()
            return self.redis_client.zincrby(name, increment)

        except:
            log.info('{}  执行失败'.format('Redis z_increase_member'))

    # 计算给定的一个或多个有序集的交集，其中给定 key 的数量必须以 numkeys 参数指定，并将该交集(结果集)储存到 destination
    def z_integrate_score(self, dest, keys, aggregate=None):
        try:
            self.re_connect()
            return self.redis_client.zinterstore(dest, keys, aggregate=aggregate)

        except:
            log.info('{}  执行失败'.format('Redis z_integrate_score'))

    # 计算给定的一个或多个有序集的并集，其中给定 key 的数量必须以 numkeys 参数指定，并将该交集(结果集)储存到 destination
    def z_union_score(self, dest, keys, aggregate=None):
        try:
            self.re_connect()
            return self.redis_client.zunionstore(dest, keys, aggregate=aggregate)

        except:
            log.info('{}  执行失败'.format('Redis z_union_score'))


    # 在有序集合中计算指定字典区间内成员数量
    def z_get_count_of_members_by_lexicographical_order(self, key, min, max):
        try:
            self.re_connect()
            return self.redis_client.zlexcount(key, min, max)

        except:
            log.info('{}  执行失败'.format('Redis z_get_count_of_members_by_lexicographical_order'))


    # 通过字典区间返回有序集合的成员
    def z_get_members_by_lexicographical_order(self, key, min, max, start=None, num=None):
        try:
            self.re_connect()
            return self.redis_client.zrangebylex(key, min, max, start, num)

        except:
            log.info('{}  执行失败'.format('Redis z_get_members_by_lexicographical_order'))


    # 通过分数返回有序集合指定区间内的成员
    def z_get_members_by_score(self, name, min, max, start=None, num=None, withscores=False):
        try:
            self.re_connect()
            return self.redis_client.zrangebyscore(name, min, max, start=start, num=num, withscores=withscores)
        except:
            log.info('{}  执行失败'.format('Redis z_get_members_by_score'))


    # 返回有序集合中指定成员的索引
    def z_get_rank_of_member(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.zrank(name, value)

        except:
            log.info('{}  执行失败'.format('Redis z_get_rank_of_member'))

    # 移除有序集合中的一个或多个成员
    def z_delete_members(self, name, *values):
        try:
            self.re_connect()
            return self.redis_client.zrem(name, *values)

        except:
            log.info('{}  执行失败'.format('Redis z_delete_members'))

    # 移除有序集合中给定的字典区间的所有成员
    def z_delete_members_by_lexicographical_order(self, key, min, max, start=None, num=None):
        try:
            self.re_connect()
            return self.redis_client.zremrangebylex(key, min, max)

        except:
            log.info('{}  执行失败'.format('Redis z_delete_members_by_lexicographical_order'))

    # 移除有序集合中给定的排名区间的所有成员
    def z_delete_members_by_rank(self, key, min, max):
        try:
            self.re_connect()
            return self.redis_client.zremrangebyrank(key, min, max)

        except:
            log.info('{}  执行失败'.format('Redis z_delete_members_by_rank'))

    # 移除有序集合中给定的分数区间的所有成员
    def z_delete_members_by_score(self, key, min, max):
        try:
            self.re_connect()
            return self.redis_client.zremrangebyscore(key, min, max)

        except:
            log.info('{}  执行失败'.format('Redis z_delete_members_by_score'))

    # 通过字典区间返回倒序排列后的有序集合成员
    def z_get_reverse_members_by_lexicographical_order(self, key, max, min, start=None, num=None):
        try:
            self.re_connect()
            return self.redis_client.zrevrangebylex(key, max, min, start, num)

        except:
            log.info('{}  执行失败'.format('Redis z_get_reverse_members_by_lexicographical_order'))


    # 通过分数返回倒序排列后有序集合指定区间内的成员
    def z_get_reverse_members_by_score(self, name, max, min, start=None, num=None, withscores=False):
        try:
            self.re_connect()
            return self.redis_client.zrevrangebyscore(name, max, min, start=start, num=num, withscores=withscores)
        except:
            log.info('{}  执行失败'.format('Redis z_get_reverse_members_by_score'))


    # 返回倒序排列后有序集合中指定成员的索引
    def z_get_reverse_rank_of_member(self, name, value):
        try:
            self.re_connect()
            return self.redis_client.zrevrank(name, value)

        except:
            log.info('{}  执行失败'.format('Redis z_get_reverse_rank_of_member'))


    # 通过索引区间返回倒序排列后有序集合成指定区间内的成员
    def z_get_reverse_members_by_index(self, name, start, stop, withscores=False):
        try:
            self.re_connect()
            return self.redis_client.zrevrange(name, start, stop, withscores=withscores)

        except:
            log.info('{}  执行失败'.format('Redis z_get_reverse_members_by_index'))


    # 通过索引区间返回有序集合成指定区间内的成员
    def z_get_members_by_index(self, name, start, stop, withscores=False):
        try:
            self.re_connect()
            return self.redis_client.zrange(name, start, stop, withscores=withscores)

        except:
            log.info('{}  执行失败'.format('Redis z_get_members_by_index'))

    # 获取成员的分数
    def z_get_score_of_members(self, set_name, key_name):
        try:
            self.re_connect()
            return self.redis_client.zscore(set_name, key_name)

        except:
            log.info('{}  执行失败'.format('Redis z_get_score_of_members'))

    # 获取分数最小的成员
    def z_get_min_score(self, set_name):
        try:
            self.re_connect()
            return int(self.redis_client.zrange(name=set_name, withscores=True, start=0, end=0)[0][1])

        except:
            log.info('{}  执行失败'.format('Redis z_get_min_score'))


    # 获取分数最大的成员
    def z_get_max_score(self, set_name):
        try:
            self.re_connect()
            return int(self.redis_client.zrevrange(name=set_name, withscores=True, start=0, end=0)[0][1])

        except:
            log.info('{}  执行失败'.format('Redis z_get_max_score'))


    # 获取成员数量
    def z_get_length_of_zset(self, set_name):
        try:
            self.re_connect()
            return self.redis_client.zcard(set_name)

        except:
            log.info('{}  执行失败'.format('Redis z_get_length_of_zset'))


    '''
    Web Socket
    '''
    # 发布数据
    def publish(self, channel, message):
        try:
            self.re_connect()
            return self.redis_client.publish(channel, message)

        except Exception as e:
            log.info('{}  执行失败  {}'.format('Redis publish', e))

    # 订阅
    def pubsub(self):
        try:
            self.re_connect()
            return self.redis_client.pubsub()
        except Exception as e:
            log.info('{}  执行失败 {}'.format('Redis pubsub', e))