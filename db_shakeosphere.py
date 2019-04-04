
import psycopg2 as ps

def assert_role(role):
    assert(role in [ 'author', 'printer', 'publisher', 'bookseller' ])

def get_indivs(cursor, role, year_low, year_high):
    # IMPORTANT: Since we are manually inserting the role into the SQL
    # query, make sure it is actually a role, to prevent a SQL injection.
    assert_role(role)
        
    cursor.execute("""
    select person.pid, first_name, last_name, count(*)
    from estc.pub_year, navigation.person, navigation.person_effective,
         navigation.""" + role + """
    where pubdate >= %(year_low)s
      and pubdate <= %(year_high)s
      and pub_year.id = """ + role + """.id
      and """ + role + """.pid = person_effective.effective_id
      and person_effective.pid = person.pid
    group by 1,2,3;""",
    { 'year_low':year_low, 'year_high':year_high })
    return cursor.fetchall()

def get_publications(cursor, year_low, year_high):
    cursor.execute("""
    select publication.id, title, pubdate
    from estc.publication, estc.pub_year
    where publication.id = pub_year.id
      and pubdate >= %(year_low)s
      and pubdate <= %(year_high)s;""",
    { 'year_low':year_low, 'year_high':year_high })
    return cursor.fetchall()

def get_person_to_pub(cursor):
    cursor.execute(""" select * from navigation.all_roles """)
    return cursor.fetchall()

def connect(user='bgreteman', password='stanford16'):
#connect(user='bhie', password='estc2016'):
    conn = ps.connect(user=user, password=password,
                      host='marengo.info-science.uiowa.edu',
                      database='estc')
    cursor = conn.cursor()
    return cursor, conn

if __name__ == '__main__':
    # Simple test.
    cursor, conn = connect()
    cursor.execute("""
    select publication.id, title
    from estc.publication, estc.pub_year
    where publication.id = pub_year.id
      and pubdate >= 1610
      and pubdate <= 1612;""")
    print(cursor.fetchall())
    cursor.close()
    conn.close()
