import bcipy

def main():
    session = bcipy.Session.create()
    session.verify()
    print(session)

if __name__ == "__main__":
    main()