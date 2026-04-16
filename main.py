from hybrid_gs.pipeline import main


if __name__ == "__main__":
    # Keep the top-level entrypoint minimal so all training logic stays in
    # `hybrid_gs.pipeline`. That makes it easier to import the package from
    # tests or future scripts without duplicating argument parsing here.
    main()
